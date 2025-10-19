import os
import time
import math
import logging
import random
from datetime import datetime, timezone, timedelta
from collections import deque

import ccxt
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
from ccxt.base.errors import DDoSProtection, ExchangeNotAvailable, RateLimitExceeded

load_dotenv()

# ====================== Config ======================
EXCHANGE_ID = os.getenv("EXCHANGE", "kraken")
QUOTE_ASSET  = os.getenv("QUOTE_ASSET", "USD").upper()

# Sell when profit >= TARGET_PCT minus estimated round-trip fees.
# We implement this conservatively as: price >= avg_cost * (1 + TARGET_PCT + 2*FEE_RATE)
TARGET_PCT   = float(os.getenv("TARGET_PCT", "0.05"))      # 5% target
FEE_RATE     = os.getenv("FEE_RATE", "0.004")  # default 0.40% taker; if "", attempt live fetch per pair
FEE_RATE     = None if FEE_RATE.strip() == "" else float(FEE_RATE)

SELL_FRACTION     = float(os.getenv("SELL_FRACTION", "1.0"))   # 1.0 = sell 100% of free units when target hit
MIN_USD_ORDER     = float(os.getenv("MIN_USD_ORDER", "5"))
MAX_SYMBOLS_PER_RUN = int(os.getenv("MAX_SYMBOLS_PER_RUN", "100"))
INTERVAL_SEC      = int(os.getenv("INTERVAL_SEC", "900"))  # 15m
HISTORY_DAYS      = int(os.getenv("HISTORY_DAYS", "180"))  # how deep to look for cost basis
MAX_TRADES_PULL   = int(os.getenv("MAX_TRADES_PULL", "2000"))  # per symbol cap

# Execution
DRY_RUN = os.getenv("DRY_RUN", "1") == "1"

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
VERBOSE_SYMBOL_LOG = os.getenv("VERBOSE_SYMBOL_LOG", "1") == "1"

API_KEY    = os.getenv("API_KEY", "")
API_SECRET = os.getenv("API_SECRET", "")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)
log = logging.getLogger("kraken-seller-bot")
log.info("=== Seller Bot boot ===")
log.info(
    f"Config: EXCHANGE={EXCHANGE_ID} QUOTE_ASSET={QUOTE_ASSET} DRY_RUN={DRY_RUN} "
    f"INTERVAL_SEC={INTERVAL_SEC} TARGET_PCT={TARGET_PCT} FEE_RATE={FEE_RATE} "
    f"SELL_FRACTION={SELL_FRACTION} MIN_USD_ORDER={MIN_USD_ORDER} "
    f"HISTORY_DAYS={HISTORY_DAYS} MAX_TRADES_PULL={MAX_TRADES_PULL} "
    f"MAX_SYMBOLS_PER_RUN={MAX_SYMBOLS_PER_RUN} LOG_LEVEL={LOG_LEVEL} "
    f"VERBOSE_SYMBOL_LOG={VERBOSE_SYMBOL_LOG}"
)

# ====================== Helpers ======================
STABLE_LIKE = {
    "USD","USDT","USDC","DAI","EUR","GBP","TUSD","FDUSD","PYUSD","BUSD","GUSD","USDP","RLUSD","EURC","USDD","USDE"
}

def is_stablecoin(asset: str) -> bool:
    return asset.upper() in STABLE_LIKE

def set_verbose_public(exchange: ccxt.Exchange, enabled: bool):
    # NEVER enable verbose for private calls (would print headers)
    try:
        exchange.verbose = bool(enabled)
    except Exception:
        pass

def make_exchange() -> ccxt.Exchange:
    klass = getattr(ccxt, EXCHANGE_ID)
    ex = klass({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {"adjustForTimeDifference": True},
    })
    ex.verbose = False
    return ex

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
def fetch_ticker(exchange, symbol):
    set_verbose_public(exchange, LOG_LEVEL == "DEBUG")
    try:
        return exchange.fetch_ticker(symbol)
    finally:
        set_verbose_public(exchange, False)
        time.sleep(0.03 + random.random()*0.05)

def try_fetch_pair_fee_rate(exchange, symbol) -> float | None:
    """
    Try to get your actual taker fee for this pair using Kraken /private/TradeVolume.
    Returns None on failure.
    """
    # DO NOT enable verbose for private
    try:
        if hasattr(exchange, "privatePostTradeVolume"):
            # Kraken wants 'pair' like 'XXBTZUSD' (exchange id), but ccxt usually accepts 'pair': symbol.id
            m = exchange.market(symbol)
            pair_id = m.get("id", None)
            payload = {"pair": pair_id} if pair_id else {}
            resp = exchange.privatePostTradeVolume(payload)
            # Kraken returns fees for taker in 'fees' and maker in 'fees_maker'. We'll take taker.
            fees = resp.get("result", {}).get("fees", {})
            if fees:
                # fees looks like { "XXBTZUSD": {"fee": "0.26", "minfee": "0.0", ... } } percentages
                fee_info = fees.get(pair_id) or next(iter(fees.values()))
                fee_pct = float(fee_info.get("fee", "0.26"))  # percent, not decimal
                return fee_pct / 100.0
        return None
    except Exception as e:
        if LOG_LEVEL == "DEBUG":
            log.debug(f"TradeVolume fee fetch failed for {symbol}: {e}")
        return None

def market_symbol_for_base(exchange, base: str, quote: str) -> str | None:
    # Try BASE/QUOTE first, then QUOTE markets that map to same base
    sym = f"{base}/{quote}"
    if sym in exchange.markets:
        m = exchange.market(sym)
        if m.get("spot", False) and m.get("active", True):
            return sym
    # Some assets use alt codes; fall back by scanning markets
    for s, m in exchange.markets.items():
        if m.get("spot", False) and m.get("active", True) and m.get("base") == base and m.get("quote") == quote:
            return s
    return None

def fifo_cost_basis_from_trades(trades, remaining_amount) -> float | None:
    """
    Compute avg cost for the *remaining* long position using FIFO over fills.
    Trades: ccxt unified my_trades in chronological order.
    We only consider 'buy' to add inventory and 'sell' to remove.
    Returns avg_cost in QUOTE_ASSET per 1 BASE, or None if cannot compute.
    """
    if remaining_amount <= 0:
        return None
    # Build FIFO queue of (qty, price)
    q = deque()
    for t in trades:
        side  = t.get("side")
        price = float(t.get("price", 0) or 0)
        amount= float(t.get("amount", 0) or 0)
        if amount <= 0 or price <= 0:
            continue
        if side == "buy":
            q.append([amount, price])
        elif side == "sell":
            # reduce from queue
            qty_to_remove = amount
            while qty_to_remove > 0 and q:
                lot = q[0]
                take = min(lot[0], qty_to_remove)
                lot[0] -= take
                qty_to_remove -= take
                if lot[0] <= 1e-15:
                    q.popleft()
            # if sells exceeded buys, ignore negative (no long position)
    # Aggregate remaining lots until remaining_amount covered
    covered, total_cost = 0.0, 0.0
    for amt, px in q:
        if covered >= remaining_amount:
            break
        take = min(amt, remaining_amount - covered)
        covered += take
        total_cost += take * px
    if covered <= 0:
        return None
    return total_cost / covered

def safe_fetch_my_trades(exchange, symbol, since_ms, limit=1000):
    """
    Pull user trades with retries and rate-limit handling.
    """
    out = []
    params = {}
    cursor_since = since_ms
    while len(out) < MAX_TRADES_PULL:
        try:
            batch = exchange.fetch_my_trades(symbol, since=cursor_since, limit=min(1000, MAX_TRADES_PULL-len(out)), params=params)
            if not batch:
                break
            out.extend(batch)
            # Advance since by last trade timestamp + 1
            last_ts = int(batch[-1]["timestamp"])
            cursor_since = last_ts + 1
            # Small jitter
            time.sleep(0.05 + random.random()*0.05)
            if len(batch) < 500:
                break
        except (DDoSProtection, RateLimitExceeded):
            log.warning(f"Rate-limited fetching trades for {symbol}; backing off 30s…")
            time.sleep(30)
        except ExchangeNotAvailable as e:
            log.warning(f"Exchange unavailable while fetching trades for {symbol}: {e}; 30s backoff")
            time.sleep(30)
        except Exception as e:
            log.warning(f"fetch_my_trades error for {symbol}: {e}")
            break
    # Ensure chronological order
    out.sort(key=lambda t: t.get("timestamp", 0))
    return out

def select_sell_candidates(exchange, balances):
    """
    From balances, select non-stable assets with free > 0 and with a BASE/QUOTE market.
    Returns list of dicts with keys: base, symbol, free
    """
    cands = []
    count = 0
    for base, amt in balances.items():
        if count >= MAX_SYMBOLS_PER_RUN:
            break
        if is_stablecoin(base):
            continue
        free_amt = float(amt or 0.0)
        if free_amt <= 0:
            continue
        sym = market_symbol_for_base(exchange, base, QUOTE_ASSET)
        if sym:
            cands.append({"base": base, "symbol": sym, "free": free_amt})
            count += 1
    return cands

def get_quote_notional(exchange, symbol, amount) -> float:
    t = fetch_ticker(exchange, symbol)
    price = float(t.get("last") or 0.0)
    return price * amount, price

def required_threshold(entry_price, fee_rate, target_pct):
    """
    Conservative profit requirement: cover maker/taker uncertainty by assuming taker on both sides.
    Threshold = entry * (1 + target_pct + 2*fee_rate)
    """
    return entry_price * (1.0 + target_pct + 2.0 * fee_rate)

def place_market_sell(exchange, symbol, amount):
    # Private path: keep verbose OFF
    try:
        m = exchange.market(symbol)
        amt_prec = exchange.amount_to_precision(symbol, amount)
        if DRY_RUN:
            log.info(f"[DRY_RUN] Would SELL {symbol} amount={amt_prec} (market)")
            return {"id": "dry-run", "symbol": symbol, "amount": amt_prec}
        order = exchange.create_order(symbol=symbol, type="market", side="sell", amount=amt_prec)
        log.info(f"Sell order placed: {order}")
        return order
    except Exception as e:
        log.exception(f"Sell failed for {symbol}: {e}")
        return None

# ====================== Main ======================
def main_loop():
    ex = make_exchange()
    log.info(f"Connected to exchange: {EXCHANGE_ID} (rateLimit={getattr(ex, 'rateLimit', 'n/a')} ms)")
    ex.load_markets()

    while True:
        try:
            # --- Fetch balances (private; keep verbose OFF) ---
            try:
                bal = ex.fetch_balance()
            except (DDoSProtection, RateLimitExceeded):
                log.warning("Rate-limited fetching balance; backing off 60s…")
                time.sleep(60)
                continue
            except ExchangeNotAvailable as e:
                log.warning(f"Exchange unavailable on balance: {e}; 60s backoff")
                time.sleep(60)
                continue

            # Map: base asset -> free amount
            free_map = {k: float(v) for k, v in (bal.get("free") or {}).items()}
            if not free_map:
                # Some setups only expose 'total'
                free_map = {k: float(v) for k, v in (bal.get("total") or {}).items()}

            candidates = select_sell_candidates(ex, free_map)
            if not candidates:
                log.info("No non-stable, non-zero balances to evaluate.")
            else:
                log.info(f"Evaluating up to {len(candidates)} positions for profit-taking (quote={QUOTE_ASSET})")

            for item in candidates:
                base   = item["base"]
                symbol = item["symbol"]
                free   = item["free"]

                # Determine pair fee to use
                fee_used = FEE_RATE
                if fee_used is None:
                    fee_live = try_fetch_pair_fee_rate(ex, symbol)
                    fee_used = fee_live if (fee_live is not None) else 0.004  # fallback 0.40%
                    if fee_live is not None:
                        log.debug(f"[{symbol}] using live taker fee {fee_used*100:.3f}%")
                    else:
                        log.debug(f"[{symbol}] using fallback taker fee {fee_used*100:.3f}%")

                # Pull trades for HISTORY_DAYS to estimate cost basis
                since = int((datetime.now(timezone.utc) - timedelta(days=HISTORY_DAYS)).timestamp() * 1000)
                trades = safe_fetch_my_trades(ex, symbol, since_ms=since, limit=min(1000, MAX_TRADES_PULL))

                # If we didn’t get trades, skip (can’t compute entry)
                if not trades:
                    log.debug(f"[{symbol}] no trades found in last {HISTORY_DAYS} days; skipping.")
                    continue

                # Compute avg cost for remaining long using FIFO-ish approach
                avg_cost = fifo_cost_basis_from_trades(trades, remaining_amount=free)
                if avg_cost is None or avg_cost <= 0:
                    log.debug(f"[{symbol}] could not compute cost basis; skipping.")
                    continue

                # Current price + threshold
                notional_now, last_px = get_quote_notional(ex, symbol, free)
                if last_px <= 0:
                    log.debug(f"[{symbol}] invalid last price; skipping.")
                    continue

                thresh_px = required_threshold(avg_cost, fee_used, TARGET_PCT)
                margin = (last_px / avg_cost) - 1.0

                if VERBOSE_SYMBOL_LOG:
                    log.info(
                        f"[{symbol}] free={free:.10g} avg_cost={avg_cost:.10g} last={last_px:.10g} "
                        f"margin={margin*100:.2f}% threshold_px={thresh_px:.10g} "
                        f"fee_used={fee_used*100:.2f}%"
                    )

                if last_px >= thresh_px:
                    # Amount to sell (respect precision)
                    amount_to_sell = free * SELL_FRACTION
                    # Ensure order meets minimum notional
                    est_notional = amount_to_sell * last_px
                    if est_notional < MIN_USD_ORDER:
                        log.info(f"[{symbol}] Hit target but notional {est_notional:.2f} < MIN_USD_ORDER {MIN_USD_ORDER:.2f}; skip.")
                        continue
                    log.info(f"[{symbol}] Profit target met → SELL {amount_to_sell:.10g} (≈ {est_notional:.2f} {QUOTE_ASSET})")
                    place_market_sell(ex, symbol, amount_to_sell)
                else:
                    if VERBOSE_SYMBOL_LOG:
                        shortfall = (thresh_px / last_px) - 1.0
                        log.debug(f"[{symbol}] target not met; needs +{shortfall*100:.2f}% more price.")
        except Exception as e:
            log.exception("Top-level loop error")

        log.info(f"Sleeping {INTERVAL_SEC}s…")
        time.sleep(INTERVAL_SEC)

if __name__ == "__main__":
    main_loop()
