# ============================================================
# SMT ALERT BOT
# Coins: BTC, ETH, SOL, BNB  |  Timeframes: 15m, 1H
# Comparison: Synthetic BTC Dominance from Binance.US data
# ALL data from Binance.US — no external APIs needed
# Scan: every 5 min | BTC.D refresh: every 15 min
# ============================================================

import os
import time
import logging
import threading
import requests
import pandas as pd
from datetime import datetime, timezone
from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
import telebot

# ── Config ───────────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '')
CHAT_ID        = os.environ.get('CHAT_ID', '')

# ── Coins to monitor ─────────────────────────────────────────
COINS = {
    'BTC': 'BTCUSDT',
    'ETH': 'ETHUSDT',
    'SOL': 'SOLUSDT',
    'BNB': 'BNBUSDT',
}

# ── Dominance basket (circulating supplies — May 2025) ───────
DOMINANCE_BASKET = {
    'BTCUSDT':  19_700_000,
    'ETHUSDT':  120_270_000,
    'BNBUSDT':  140_890_000,
    'SOLUSDT':  465_000_000,
    'XRPUSDT':  57_800_000_000,
    'ADAUSDT':  35_600_000_000,
    'AVAXUSDT': 410_000_000,
    'DOGEUSDT': 146_000_000_000,
    'MATICUSDT':9_900_000_000,
    'DOTUSDT':  1_430_000_000,
}

COMP_LABEL = 'BTC.D'

# ── Timeframes ───────────────────────────────────────────────
TIMEFRAMES = {
    '15m': '15m',
    '1h':  '1h',
}

# ── API ──────────────────────────────────────────────────────
BINANCE_US_BASE  = 'https://api.binance.us/api/v3'
BINANCE_INTERVAL = {'15m': '15m', '1h': '1h'}

# ── SMT Settings ─────────────────────────────────────────────
PIVOT_LOOKBACK   = 1
PIVOT_A_STRENGTH = 2
SYNC_TOL         = 2
MAX_SIGNAL_SPAN  = 180
CROSS_TOL_PCT    = 0.02

# ── Timing ───────────────────────────────────────────────────
COOLDOWN_SECONDS     = 30 * 60   # 30 min alert cooldown
BTCD_REFRESH_MINUTES = 15        # refresh BTC.D every 15 min

# ── BTC.D Cache ──────────────────────────────────────────────
BTCD_CACHE:    dict = {}          # tf -> DataFrame
BTCD_CACHE_TS: dict = {}          # tf -> epoch float
BTCD_LOCK = threading.Lock()

# ── Alert state ──────────────────────────────────────────────
last_alerts     = {}
sent_signatures = set()

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)
app    = Flask(__name__)

try:
    bot = telebot.TeleBot(TELEGRAM_TOKEN, threaded=False)
    logger.info("Telegram bot initialized")
except Exception as e:
    logger.error(f"Telegram init error: {e}")
    bot = None


# ============================================================
# BINANCE.US — OHLC FETCH
# ============================================================

def fetch_klines(symbol: str, interval: str,
                 limit: int = 500) -> pd.DataFrame | None:
    """Fetch OHLC candles from Binance.US."""
    try:
        resp = requests.get(
            f"{BINANCE_US_BASE}/klines",
            params={
                'symbol':   symbol,
                'interval': interval,
                'limit':    limit,
            },
            timeout=15,
        )
        if resp.status_code != 200:
            logger.error(
                f"Binance.US {resp.status_code} — {symbol}/{interval}"
            )
            return None

        klines = resp.json()
        if not klines or isinstance(klines, dict):
            return None

        df = pd.DataFrame(klines, columns=[
            'time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'qav', 'num_trades',
            'taker_base', 'taker_quote', 'ignore',
        ])
        df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].astype(float)

        return df[['time', 'open', 'high', 'low', 'close']]

    except Exception as e:
        logger.error(f"fetch_klines {symbol}/{interval}: {e}")
        return None


# ============================================================
# SYNTHETIC BTC DOMINANCE
# ============================================================

def build_btcd_ohlc(interval: str) -> pd.DataFrame | None:
    """
    Build synthetic BTC Dominance OHLC from Binance.US prices.

    Formula per candle:
        total_mcap  = sum(price[symbol] × supply[symbol])
        btc_dom     = btc_price × btc_supply / total_mcap × 100

    OHLC shape follows BTC's own candle scaled by dominance ratio.
    Uses only Binance.US — zero external API calls.
    """
    try:
        bin_interval = BINANCE_INTERVAL.get(interval, '15m')

        # ── Fetch all basket symbols ──────────────────────────
        price_data: dict[str, pd.DataFrame] = {}
        for symbol in DOMINANCE_BASKET:
            df = fetch_klines(symbol, bin_interval, limit=500)
            if df is not None:
                price_data[symbol] = df.set_index('time')
            else:
                logger.warning(
                    f"BTC.D basket: could not fetch {symbol} — skipping"
                )
            time.sleep(0.2)   # gentle pacing between requests

        # ── Validate minimum basket ───────────────────────────
        if 'BTCUSDT' not in price_data:
            logger.error("BTCUSDT missing — cannot compute BTC.D")
            return None

        if len(price_data) < 3:
            logger.error(
                f"Only {len(price_data)} basket symbols fetched — aborting"
            )
            return None

        # ── Build BTC.D candle by candle ─────────────────────
        btc_df  = price_data['BTCUSDT']
        records = []

        for ts in btc_df.index:
            # Sum market caps at this timestamp using close price
            total_mcap = 0.0
            for symbol, supply in DOMINANCE_BASKET.items():
                sym_df = price_data.get(symbol)
                if sym_df is not None and ts in sym_df.index:
                    total_mcap += sym_df.loc[ts, 'close'] * supply

            if total_mcap <= 0:
                continue

            btc_supply = DOMINANCE_BASKET['BTCUSDT']

            # Scale BTC's OHLC candle shape into dominance %
            records.append({
                'time':  ts,
                'open':  (btc_df.loc[ts, 'open']  * btc_supply / total_mcap) * 100.0,
                'high':  (btc_df.loc[ts, 'high']  * btc_supply / total_mcap) * 100.0,
                'low':   (btc_df.loc[ts, 'low']   * btc_supply / total_mcap) * 100.0,
                'close': (btc_df.loc[ts, 'close'] * btc_supply / total_mcap) * 100.0,
            })

        if len(records) < 20:
            logger.error(
                f"BTC.D {interval}: only {len(records)} candles — insufficient"
            )
            return None

        ohlc = (
            pd.DataFrame(records)
            .sort_values('time')
            .reset_index(drop=True)
        )

        logger.info(
            f"✅ BTC.D {interval}: {len(ohlc)} candles | "
            f"latest = {ohlc['close'].iloc[-1]:.3f}% | "
            f"basket = {len(price_data)}/{len(DOMINANCE_BASKET)} symbols"
        )
        return ohlc

    except Exception as e:
        logger.error(f"build_btcd_ohlc {interval}: {e}")
        return None


def refresh_btcd_cache():
    """
    Rebuild BTC.D OHLC for all timeframes and update cache.
    Called by scheduler every 15 min — NOT by the scan loop.
    """
    logger.info("BTC.D cache refresh starting…")
    for tf in ['15m', '1h']:
        df = build_btcd_ohlc(tf)
        with BTCD_LOCK:
            if df is not None:
                BTCD_CACHE[tf]    = df
                BTCD_CACHE_TS[tf] = time.time()
            else:
                if tf in BTCD_CACHE:
                    logger.warning(
                        f"BTC.D {tf} refresh failed — keeping stale cache"
                    )
                else:
                    logger.warning(
                        f"BTC.D {tf} refresh failed — no data available"
                    )
        if tf == '15m':
            time.sleep(2)   # small gap between the two timeframe builds
    logger.info("BTC.D cache refresh complete")


def get_btcd(tf: str) -> pd.DataFrame | None:
    """Thread-safe read from BTC.D cache."""
    with BTCD_LOCK:
        return BTCD_CACHE.get(tf)


# ============================================================
# PIVOT DETECTION
# ============================================================

def detect_pivots(values: dict, left: int, right: int):
    highs, lows = [], []
    n = len(values['high'])
    for i in range(left, n - right):
        hi = lo = True
        for k in range(1, left + 1):
            if values['high'][i - k] >= values['high'][i]: hi = False
            if values['low'][i - k]  <= values['low'][i]:  lo = False
        for k in range(1, right + 1):
            if values['high'][i + k] >= values['high'][i]: hi = False
            if values['low'][i + k]  <= values['low'][i]:  lo = False
        if hi: highs.append((i, values['high'][i]))
        if lo: lows.append((i,  values['low'][i]))
    return highs, lows


def cross_ok(highs, lows, bar_a, px_a,
             bar_b, px_b, is_bear: bool) -> bool:
    span = bar_b - bar_a - 1
    if span <= 0:              return True
    if span > MAX_SIGNAL_SPAN: return False
    for d in range(1, span + 1):
        idx = bar_a + d
        if idx >= len(highs): return False
        line_px = px_a + (px_b - px_a) * d / (bar_b - bar_a)
        tol_px  = abs(line_px) * CROSS_TOL_PCT / 100.0
        if is_bear     and highs[idx] > line_px + tol_px: return False
        if not is_bear and lows[idx]  < line_px - tol_px: return False
    return True


def near_pivot(pivots, target: int, tol: int) -> int:
    best, best_d = -1, tol + 1
    for i, (b, _) in enumerate(pivots):
        d = abs(b - target)
        if d <= tol and d < best_d:
            best_d, best = d, i
    return best


def near_pivot_before(pivots, target: int,
                       limit: int, tol: int) -> int:
    best, best_d = -1, tol + 1
    for i, (b, _) in enumerate(pivots):
        if b >= limit: continue
        d = abs(b - target)
        if d <= tol and d < best_d:
            best_d, best = d, i
    return best


# ============================================================
# SMT DETECTION
# ============================================================

def detect_smt(coin_df: pd.DataFrame,
               comp_df: pd.DataFrame,
               comp_label: str) -> list:
    if coin_df is None or comp_df is None:      return []
    if len(coin_df) < 50 or len(comp_df) < 50: return []

    n       = min(len(coin_df), len(comp_df))
    coin_df = coin_df.tail(n).reset_index(drop=True)
    comp_df = comp_df.tail(n).reset_index(drop=True)

    cv = {
        'high': coin_df['high'].values.astype(float),
        'low':  coin_df['low'].values.astype(float),
    }
    xv = {
        'high': comp_df['high'].values.astype(float),
        'low':  comp_df['low'].values.astype(float),
    }

    cb_hi, cb_lo = detect_pivots(cv, PIVOT_LOOKBACK,   PIVOT_LOOKBACK)
    xb_hi, xb_lo = detect_pivots(xv, PIVOT_LOOKBACK,   PIVOT_LOOKBACK)
    ca_hi, ca_lo = detect_pivots(cv, PIVOT_A_STRENGTH,  PIVOT_A_STRENGTH)

    latest  = n - 1 - PIVOT_LOOKBACK
    signals = []

    # ── BEAR: coin Higher High + BTC.D Lower High ────────────
    cph = next(((b, v) for b, v in cb_hi if b == latest), None)
    if cph:
        bar_b, px_b = cph
        xi = near_pivot(xb_hi, bar_b, SYNC_TOL)
        if xi >= 0:
            xbar_b, xpx_b = xb_hi[xi]
            for bar_a, px_a in reversed(ca_hi):
                span = bar_b - bar_a
                if span <= 0:              continue
                if span > MAX_SIGNAL_SPAN: break
                xai = near_pivot_before(
                    xb_hi, bar_a, xbar_b, SYNC_TOL
                )
                if xai >= 0:
                    xbar_a, xpx_a = xb_hi[xai]
                    if px_b > px_a and xpx_b < xpx_a:
                        if (cross_ok(cv['high'], cv['low'],
                                     bar_a, px_a,
                                     bar_b, px_b, True) and
                            cross_ok(xv['high'], xv['low'],
                                     xbar_a, xpx_a,
                                     xbar_b, xpx_b, True)):
                            signals.append({
                                'direction':  'BEAR',
                                'comp_label': comp_label,
                                'coin_bar_a': bar_a,
                                'coin_px_a':  px_a,
                                'coin_bar_b': bar_b,
                                'coin_px_b':  px_b,
                                'comp_bar_a': xbar_a,
                                'comp_px_a':  xpx_a,
                                'comp_bar_b': xbar_b,
                                'comp_px_b':  xpx_b,
                                'span':       span,
                                'time_b':     coin_df['time'].iloc[bar_b],
                            })
                            break

    # ── BULL: coin Lower Low + BTC.D Higher Low ──────────────
    cpl = next(((b, v) for b, v in cb_lo if b == latest), None)
    if cpl:
        bar_b, px_b = cpl
        xi = near_pivot(xb_lo, bar_b, SYNC_TOL)
        if xi >= 0:
            xbar_b, xpx_b = xb_lo[xi]
            for bar_a, px_a in reversed(ca_lo):
                span = bar_b - bar_a
                if span <= 0:              continue
                if span > MAX_SIGNAL_SPAN: break
                xai = near_pivot_before(
                    xb_lo, bar_a, xbar_b, SYNC_TOL
                )
                if xai >= 0:
                    xbar_a, xpx_a = xb_lo[xai]
                    if px_b < px_a and xpx_b > xpx_a:
                        if (cross_ok(cv['high'], cv['low'],
                                     bar_a, px_a,
                                     bar_b, px_b, False) and
                            cross_ok(xv['high'], xv['low'],
                                     xbar_a, xpx_a,
                                     xbar_b, xpx_b, False)):
                            signals.append({
                                'direction':  'BULL',
                                'comp_label': comp_label,
                                'coin_bar_a': bar_a,
                                'coin_px_a':  px_a,
                                'coin_bar_b': bar_b,
                                'coin_px_b':  px_b,
                                'comp_bar_a': xbar_a,
                                'comp_px_a':  xpx_a,
                                'comp_bar_b': xbar_b,
                                'comp_px_b':  xpx_b,
                                'span':       span,
                                'time_b':     coin_df['time'].iloc[bar_b],
                            })
                            break

    return signals


# ============================================================
# TELEGRAM
# ============================================================

def send_msg(text: str):
    try:
        if bot is None or not CHAT_ID:
            return
        bot.send_message(CHAT_ID, text, parse_mode='HTML')
    except Exception as e:
        logger.error(f"Telegram send error: {e}")


def format_signal(coin: str, tf: str, sig: dict) -> str:
    d     = sig['direction']
    emoji = '🟢' if d == 'BULL' else '🔴'

    if d == 'BULL':
        ac = '↓ Lower Low'
        ax = '↑ Higher Low  ← BTC.D DIVERGED'
        w  = '🎯 Watch for reversal UP'
        n  = ('Coin drops while BTC dominance rises → '
              'altcoin underperforming → reversal UP expected')
    else:
        ac = '↑ Higher High'
        ax = '↓ Lower High  ← BTC.D DIVERGED'
        w  = '🎯 Watch for reversal DOWN'
        n  = ('Coin pumps while BTC dominance falls → '
              'altcoin overperforming → reversal DOWN expected')

    ts = pd.Timestamp(sig['time_b']).strftime('%H:%M UTC %d-%b')

    return (
        f"{emoji} <b>{d} SMT — {coin}/USDT {tf}</b>\n"
        f"{'─'*32}\n"
        f"<b>Coin Pivot A:</b>   ${sig['coin_px_a']:,.4f}  "
        f"({sig['span']} bars ago)\n"
        f"<b>Coin Pivot B:</b>   ${sig['coin_px_b']:,.4f}  (now)\n"
        f"<b>Coin Move:</b>      {ac}\n\n"
        f"<b>BTC.D Pivot A:</b>  {sig['comp_px_a']:.3f}%\n"
        f"<b>BTC.D Pivot B:</b>  {sig['comp_px_b']:.3f}%\n"
        f"<b>BTC.D Move:</b>     {ax}\n\n"
        f"<b>Span:</b>  {sig['span']} bars\n"
        f"<b>Time:</b>  {ts}\n\n"
        f"<i>{n}</i>\n\n"
        f"{w}"
    )


def send_startup_msg():
    basket_str = ', '.join(
        s.replace('USDT', '') for s in DOMINANCE_BASKET
    )
    send_msg(
        f"🤖 <b>SMT Alert Bot — LIVE</b>\n"
        f"{'─'*32}\n"
        f"<b>Coins:</b>       {', '.join(COINS)}\n"
        f"<b>Timeframes:</b>  {', '.join(TIMEFRAMES)}\n"
        f"<b>Comparison:</b>  Synthetic BTC Dominance (BTC.D)\n\n"
        f"<b>Data source:</b> Binance.US only\n"
        f"<b>BTC.D basket:</b> {basket_str}\n\n"
        f"<b>SMT Logic:</b>\n"
        f"  🟢 BULL = Coin Lower Low   + BTC.D Higher Low\n"
        f"  🔴 BEAR = Coin Higher High + BTC.D Lower High\n\n"
        f"<b>Scan:</b>         Every 5 minutes\n"
        f"<b>BTC.D refresh:</b> Every 15 minutes\n"
        f"<b>Alert cooldown:</b> 30 minutes\n\n"
        f"<i>Bot is live and scanning…</i>"
    )


# ============================================================
# SCAN LOOP
# ============================================================

def scan_all():
    try:
        now = datetime.now(timezone.utc)
        logger.info(f"🔍 Scan at {now.strftime('%H:%M:%S UTC')}")

        for coin_name, coin_ticker in COINS.items():
            for tf_label, tf_interval in TIMEFRAMES.items():

                # Read BTC.D from cache — never fetches here
                comp_df = get_btcd(tf_label)
                if comp_df is None:
                    logger.warning(
                        f"BTC.D not ready for {tf_label} — "
                        f"skipping {coin_name}"
                    )
                    continue

                # Fetch coin OHLC from Binance.US
                coin_df = fetch_klines(
                    coin_ticker,
                    BINANCE_INTERVAL[tf_interval],
                    limit=500,
                )
                if coin_df is None:
                    logger.warning(
                        f"No Binance data for {coin_name} {tf_label}"
                    )
                    continue

                # Run SMT detection
                for sig in detect_smt(coin_df, comp_df, COMP_LABEL):

                    # Deduplication key
                    sig_key = (
                        f"{coin_name}_{tf_label}_{sig['direction']}_"
                        f"{sig['coin_bar_a']}_{sig['coin_bar_b']}_"
                        f"{round(sig['coin_px_a'], 4)}_"
                        f"{round(sig['coin_px_b'], 4)}"
                    )
                    if sig_key in sent_signatures:
                        continue

                    # Cooldown check
                    ck = (f"{coin_name}_{tf_label}_"
                          f"{sig['direction']}_BTCD")
                    if (time.time() - last_alerts.get(ck, 0)
                            < COOLDOWN_SECONDS):
                        continue

                    # Send alert
                    send_msg(format_signal(coin_name, tf_label, sig))
                    sent_signatures.add(sig_key)
                    last_alerts[ck] = time.time()
                    logger.info(
                        f"✅ {sig['direction']} SMT {coin_name} "
                        f"{tf_label} | "
                        f"coin={sig['coin_px_b']:.4f} "
                        f"btcd={sig['comp_px_b']:.3f}%"
                    )

                time.sleep(0.2)   # pace Binance calls

        # Trim old signatures
        if len(sent_signatures) > 2000:
            for s in list(sent_signatures)[:1000]:
                sent_signatures.discard(s)

        logger.info(
            f"✅ Scan complete — "
            f"{len(sent_signatures)} signals sent so far"
        )

    except Exception as e:
        logger.error(f"scan_all error: {e}")


# ============================================================
# FLASK ROUTES
# ============================================================

@app.route('/')
def index():
    now  = datetime.now(timezone.utc)
    rows = ''
    for tf in TIMEFRAMES:
        df  = get_btcd(tf)
        age = int(time.time() - BTCD_CACHE_TS.get(tf, 0))
        if df is not None:
            rows += (
                f"<li><b>{tf}</b>: {len(df)} candles | "
                f"latest BTC.D = {df['close'].iloc[-1]:.3f}% | "
                f"cache age = {age}s</li>"
            )
        else:
            rows += (
                f"<li><b>{tf}</b>: "
                f"building… (first load ~30s)</li>"
            )
    coins_html  = ''.join(f"<li>{c}/USDT</li>" for c in COINS)
    basket_html = ''.join(
        f"<li>{s.replace('USDT', '')}</li>"
        for s in DOMINANCE_BASKET
    )
    return (
        f"<h2>🤖 SMT Alert Bot</h2>"
        f"<p><b>Status:</b> ✅ Running</p>"
        f"<p><b>Time UTC:</b> {now.strftime('%Y-%m-%d %H:%M:%S')}</p>"
        f"<p><b>Data source:</b> Binance.US only</p>"
        f"<p><b>Scan:</b> every 5 min | "
        f"<b>BTC.D refresh:</b> every 15 min</p>"
        f"<p><b>Total alerts sent:</b> {len(sent_signatures)}</p>"
        f"<p><b>Active cooldowns:</b> {len(last_alerts)}</p>"
        f"<h3>Monitoring:</h3><ul>{coins_html}</ul>"
        f"<h3>BTC.D Cache:</h3><ul>{rows}</ul>"
        f"<h3>Dominance Basket:</h3><ul>{basket_html}</ul>"
        f"<p>"
        f"<a href='/btcd'>View BTC.D candles</a> | "
        f"<a href='/scan_now'>Force scan</a> | "
        f"<a href='/refresh_btcd'>Force BTC.D refresh</a> | "
        f"<a href='/status'>JSON status</a>"
        f"</p>"
    )


@app.route('/health')
def health():
    return 'OK', 200


@app.route('/status')
def status():
    info = {}
    for tf in TIMEFRAMES:
        df = get_btcd(tf)
        info[tf] = {
            'candles': len(df) if df is not None else 0,
            'latest':  float(df['close'].iloc[-1]) if df is not None else None,
            'age_sec': int(time.time() - BTCD_CACHE_TS.get(tf, 0)),
        }
    return {
        'status':      'running',
        'time_utc':    datetime.now(timezone.utc).isoformat(),
        'alerts_sent': len(sent_signatures),
        'cooldowns':   len(last_alerts),
        'btcd_cache':  info,
        'basket_size': len(DOMINANCE_BASKET),
    }


@app.route('/btcd')
def show_btcd():
    out = []
    for tf in TIMEFRAMES:
        df = get_btcd(tf)
        if df is not None:
            out.append(
                f"<h3>Synthetic BTC.D — {tf} "
                f"(last 10 candles)</h3>"
                f"<pre>{df.tail(10).to_string(index=False)}</pre>"
            )
        else:
            out.append(
                f"<h3>BTC.D {tf}</h3>"
                f"<p>Not loaded yet — check back in 30s.</p>"
            )
    return ''.join(out) or '<p>No BTC.D data yet.</p>'


@app.route('/scan_now')
def scan_now_route():
    threading.Thread(target=scan_all, daemon=True).start()
    return 'Scan triggered! Check logs.', 200


@app.route('/refresh_btcd')
def refresh_btcd_route():
    threading.Thread(
        target=refresh_btcd_cache, daemon=True
    ).start()
    return 'BTC.D refresh triggered! Check /btcd in ~30s.', 200


@app.route('/reset')
def reset():
    sent_signatures.clear()
    last_alerts.clear()
    with BTCD_LOCK:
        BTCD_CACHE.clear()
        BTCD_CACHE_TS.clear()
    return 'Reset done — all state cleared.', 200


# ============================================================
# STARTUP
# ============================================================

def start_bot():
    logger.info("SMT Bot starting — synthetic BTC.D / Binance.US only")

    try:
        send_startup_msg()
    except Exception as e:
        logger.error(f"Startup msg error: {e}")

    # Build BTC.D cache immediately in background
    threading.Thread(
        target=refresh_btcd_cache, daemon=True
    ).start()

    scheduler = BackgroundScheduler(timezone='UTC')

    # ── Scan every 5 minutes ─────────────────────────────────
    scheduler.add_job(
        scan_all,
        trigger='cron',
        minute='0,5,10,15,20,25,30,35,40,45,50,55',
        id='scan_job',
        max_instances=1,
        coalesce=True,
        misfire_grace_time=30,
    )

    # ── Refresh BTC.D every 15 minutes ───────────────────────
    scheduler.add_job(
        refresh_btcd_cache,
        trigger='cron',
        minute='0,15,30,45',
        id='btcd_refresh_job',
        max_instances=1,
        coalesce=True,
        misfire_grace_time=60,
    )

    scheduler.start()
    logger.info(
        "Scheduler started — "
        "scan every 5 min | BTC.D refresh every 15 min"
    )


threading.Thread(target=start_bot, daemon=True).start()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
