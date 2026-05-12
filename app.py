# ============================================================
# SMT ALERT BOT
# Coins: BTC, ETH, SOL, BNB  |  Timeframes: 15m, 1H
# Comparison: BTC Dominance (BTC.D) via CoinGecko
# Crypto: Binance.US API
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

COINS = {
    'BTC': 'BTCUSDT',
    'ETH': 'ETHUSDT',
    'SOL': 'SOLUSDT',
    'BNB': 'BNBUSDT',
}

COMP_LABEL = 'BTC.D'

TIMEFRAMES = {
    '15m': '15m',
    '1h':  '1h',
}

BINANCE_US_BASE = 'https://api.binance.us/api/v3'
COINGECKO_BASE  = 'https://api.coingecko.com/api/v3'

BINANCE_INTERVAL = {'15m': '15m', '1h': '1h'}

# ── SMT Settings ─────────────────────────────────────────────
PIVOT_LOOKBACK   = 1
PIVOT_A_STRENGTH = 2
SYNC_TOL         = 2
MAX_SIGNAL_SPAN  = 180
CROSS_TOL_PCT    = 0.02

# ── Cooldown ─────────────────────────────────────────────────
COOLDOWN_SECONDS = 30 * 60

# ── BTC.D Cache ──────────────────────────────────────────────
# Fetched on its OWN background thread, every 15 minutes.
# Scan cycle ONLY reads from cache — never fetches directly.
BTCD_CACHE:    dict = {}   # tf -> DataFrame
BTCD_CACHE_TS: dict = {}   # tf -> epoch float
BTCD_LOCK = threading.Lock()

# How often the background fetcher runs (seconds)
BTCD_REFRESH_INTERVAL = 15 * 60   # 15 minutes

# ── Alert State ──────────────────────────────────────────────
last_alerts     = {}
sent_signatures = set()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)
app = Flask(__name__)

try:
    bot = telebot.TeleBot(TELEGRAM_TOKEN, threaded=False)
    logger.info("Telegram bot initialized")
except Exception as e:
    logger.error(f"Telegram init error: {e}")
    bot = None


# ============================================================
# BTC.D BACKGROUND FETCHER
# Runs on its own thread, every 15 minutes.
# Uses exponential backoff on 429.
# Scan loop only reads from BTCD_CACHE — zero fetch calls there.
# ============================================================

def _coingecko_get(url: str, params: dict,
                   max_retries: int = 4) -> dict | None:
    """
    GET a CoinGecko URL with exponential backoff on 429.
    Waits: 10s → 20s → 40s → 80s before giving up.
    """
    delay = 10
    for attempt in range(max_retries):
        try:
            resp = requests.get(
                url, params=params,
                timeout=25,
                headers={'Accept': 'application/json'},
            )
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                logger.warning(
                    f"CoinGecko 429 — waiting {delay}s "
                    f"(attempt {attempt+1}/{max_retries})"
                )
                time.sleep(delay)
                delay *= 2
                continue
            logger.error(
                f"CoinGecko HTTP {resp.status_code} | {url}"
            )
            return None
        except Exception as e:
            logger.error(f"CoinGecko request error: {e}")
            time.sleep(delay)
            delay *= 2
    logger.error(f"CoinGecko gave up after {max_retries} attempts: {url}")
    return None


def _build_btcd_ohlc(interval: str) -> pd.DataFrame | None:
    """
    Fetch BTC mcap + total mcap from CoinGecko,
    compute BTC.D, resample to OHLC candles.
    """
    days          = 2   if interval == '15m' else 90
    resample_rule = '15min' if interval == '15m' else '1h'
    cg_interval   = 'minutely' if interval == '15m' else 'hourly'

    # ── Call 1: BTC market cap ───────────────────────────────
    data1 = _coingecko_get(
        f"{COINGECKO_BASE}/coins/bitcoin/market_chart",
        {'vs_currency': 'usd', 'days': days, 'interval': cg_interval},
    )
    if not data1 or not data1.get('market_caps'):
        logger.error(f"BTC market_chart empty for {interval}")
        return None

    btc_df = (
        pd.DataFrame(data1['market_caps'], columns=['ts_ms', 'btc_mcap'])
        .assign(time=lambda d: pd.to_datetime(d['ts_ms'], unit='ms', utc=True))
        [['time', 'btc_mcap']]
        .set_index('time')
    )

    # ── Pause between calls ──────────────────────────────────
    time.sleep(3)

    # ── Call 2: Total crypto market cap ─────────────────────
    data2 = _coingecko_get(
        f"{COINGECKO_BASE}/global/market_cap_chart",
        {'days': days},
    )
    if not data2:
        logger.error(f"Total market_cap_chart empty for {interval}")
        return None

    raw = data2.get('market_cap_chart', {}).get('market_cap', [])
    if not raw:
        logger.error(f"Total market_cap list empty for {interval}")
        return None

    total_df = (
        pd.DataFrame(raw, columns=['ts_ms', 'total_mcap'])
        .assign(time=lambda d: pd.to_datetime(d['ts_ms'], unit='ms', utc=True))
        [['time', 'total_mcap']]
        .set_index('time')
    )

    # ── Merge, compute BTC.D, resample ──────────────────────
    merged = (
        btc_df.join(total_df, how='outer')
        .sort_index()
        .ffill()
        .dropna()
    )
    if len(merged) < 20:
        logger.error(f"BTC.D merge too short ({len(merged)}) for {interval}")
        return None

    merged['btcd'] = (merged['btc_mcap'] / merged['total_mcap']) * 100.0

    ohlc = (
        merged['btcd']
        .resample(resample_rule)
        .ohlc()
        .dropna()
        .reset_index()
    )
    ohlc.columns = ['time', 'open', 'high', 'low', 'close']
    ohlc = ohlc.sort_values('time').reset_index(drop=True)

    if len(ohlc) < 20:
        logger.error(f"BTC.D OHLC too short ({len(ohlc)}) for {interval}")
        return None

    logger.info(
        f"✅ BTC.D {interval}: {len(ohlc)} candles | "
        f"latest = {ohlc['close'].iloc[-1]:.3f}%"
    )
    return ohlc


def refresh_btcd_cache():
    """
    Refresh BTC.D OHLC for all timeframes.
    Called by its own scheduler job — NOT by the scan loop.
    Fetches 15m, waits 5s, fetches 1h.
    Total: 4 CoinGecko calls per 15 minutes = very safe.
    """
    logger.info("BTC.D cache refresh starting…")
    for tf in ['15m', '1h']:
        df = _build_btcd_ohlc(tf)
        with BTCD_LOCK:
            if df is not None:
                BTCD_CACHE[tf]    = df
                BTCD_CACHE_TS[tf] = time.time()
            else:
                logger.warning(
                    f"BTC.D {tf} refresh failed — "
                    f"{'keeping stale cache' if tf in BTCD_CACHE else 'no data yet'}"
                )
        # Gap between the two timeframe fetches
        if tf == '15m':
            time.sleep(5)
    logger.info("BTC.D cache refresh complete")


def get_btcd(tf: str) -> pd.DataFrame | None:
    """Read BTC.D from cache. Never fetches. Thread-safe."""
    with BTCD_LOCK:
        return BTCD_CACHE.get(tf)


# ============================================================
# BINANCE.US
# ============================================================

def get_ohlc_binance(symbol: str, interval: str) -> pd.DataFrame | None:
    try:
        resp = requests.get(
            f"{BINANCE_US_BASE}/klines",
            params={
                'symbol':   symbol,
                'interval': BINANCE_INTERVAL.get(interval, '15m'),
                'limit':    500,
            },
            timeout=15,
        )
        if resp.status_code != 200:
            logger.error(f"Binance.US HTTP {resp.status_code} for {symbol}")
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
        logger.error(f"Binance.US error {symbol} {interval}: {e}")
        return None


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


def cross_ok(highs, lows, bar_a, px_a, bar_b, px_b, is_bear: bool) -> bool:
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


def near_pivot_before(pivots, target: int, limit: int, tol: int) -> int:
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

    cv = {'high': coin_df['high'].values.astype(float),
          'low':  coin_df['low'].values.astype(float)}
    xv = {'high': comp_df['high'].values.astype(float),
          'low':  comp_df['low'].values.astype(float)}

    cb_hi, cb_lo = detect_pivots(cv, PIVOT_LOOKBACK,   PIVOT_LOOKBACK)
    xb_hi, xb_lo = detect_pivots(xv, PIVOT_LOOKBACK,   PIVOT_LOOKBACK)
    ca_hi, ca_lo = detect_pivots(cv, PIVOT_A_STRENGTH,  PIVOT_A_STRENGTH)

    latest  = n - 1 - PIVOT_LOOKBACK
    signals = []

    # ── BEAR ─────────────────────────────────────────────────
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
                xai = near_pivot_before(xb_hi, bar_a, xbar_b, SYNC_TOL)
                if xai >= 0:
                    xbar_a, xpx_a = xb_hi[xai]
                    if px_b > px_a and xpx_b < xpx_a:
                        if (cross_ok(cv['high'], cv['low'],
                                     bar_a, px_a, bar_b, px_b, True) and
                            cross_ok(xv['high'], xv['low'],
                                     xbar_a, xpx_a, xbar_b, xpx_b, True)):
                            signals.append({
                                'direction':  'BEAR',
                                'comp_label': comp_label,
                                'coin_bar_a': bar_a,  'coin_px_a': px_a,
                                'coin_bar_b': bar_b,  'coin_px_b': px_b,
                                'comp_bar_a': xbar_a, 'comp_px_a': xpx_a,
                                'comp_bar_b': xbar_b, 'comp_px_b': xpx_b,
                                'span': span,
                                'time_b': coin_df['time'].iloc[bar_b],
                            })
                            break

    # ── BULL ─────────────────────────────────────────────────
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
                xai = near_pivot_before(xb_lo, bar_a, xbar_b, SYNC_TOL)
                if xai >= 0:
                    xbar_a, xpx_a = xb_lo[xai]
                    if px_b < px_a and xpx_b > xpx_a:
                        if (cross_ok(cv['high'], cv['low'],
                                     bar_a, px_a, bar_b, px_b, False) and
                            cross_ok(xv['high'], xv['low'],
                                     xbar_a, xpx_a, xbar_b, xpx_b, False)):
                            signals.append({
                                'direction':  'BULL',
                                'comp_label': comp_label,
                                'coin_bar_a': bar_a,  'coin_px_a': px_a,
                                'coin_bar_b': bar_b,  'coin_px_b': px_b,
                                'comp_bar_a': xbar_a, 'comp_px_a': xpx_a,
                                'comp_bar_b': xbar_b, 'comp_px_b': xpx_b,
                                'span': span,
                                'time_b': coin_df['time'].iloc[bar_b],
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
        n  = 'Coin drops while BTC dominance rises → altcoin underperforming → reversal UP expected'
    else:
        ac = '↑ Higher High'
        ax = '↓ Lower High  ← BTC.D DIVERGED'
        w  = '🎯 Watch for reversal DOWN'
        n  = 'Coin pumps while BTC dominance falls → altcoin overperforming → reversal DOWN expected'

    ts = pd.Timestamp(sig['time_b']).strftime('%H:%M UTC %d-%b')
    return (
        f"{emoji} <b>{d} SMT — {coin}/USDT {tf}</b>\n"
        f"{'─'*32}\n"
        f"<b>Coin Pivot A:</b>   ${sig['coin_px_a']:,.4f}  ({sig['span']} bars ago)\n"
        f"<b>Coin Pivot B:</b>   ${sig['coin_px_b']:,.4f}  (now)\n"
        f"<b>Coin Move:</b>      {ac}\n\n"
        f"<b>BTC.D Pivot A:</b>  {sig['comp_px_a']:.3f}%\n"
        f"<b>BTC.D Pivot B:</b>  {sig['comp_px_b']:.3f}%\n"
        f"<b>BTC.D Move:</b>     {ax}\n\n"
        f"<b>Span:</b>  {sig['span']} bars\n"
        f"<b>Time:</b>  {ts}\n\n"
        f"<i>{n}</i>\n\n{w}"
    )


def send_startup_msg():
    send_msg(
        f"🤖 <b>SMT Alert Bot — LIVE</b>\n"
        f"{'─'*32}\n"
        f"<b>Coins:</b>       {', '.join(COINS)}\n"
        f"<b>Timeframes:</b>  {', '.join(TIMEFRAMES)}\n"
        f"<b>Comparison:</b>  BTC Dominance (BTC.D)\n\n"
        f"<b>Sources:</b>\n"
        f"  Crypto: Binance.US (live)\n"
        f"  BTC.D:  CoinGecko (refreshed every 15 min)\n\n"
        f"<b>SMT Logic:</b>\n"
        f"  🟢 BULL = Coin Lower Low  + BTC.D Higher Low\n"
        f"  🔴 BEAR = Coin Higher High + BTC.D Lower High\n\n"
        f"<b>Note:</b> First alert possible after BTC.D loads (~30s)\n\n"
        f"<i>Scanning every minute…</i>"
    )


# ============================================================
# SCAN LOOP — reads cache only, never calls CoinGecko
# ============================================================

def scan_all():
    try:
        now = datetime.now(timezone.utc)
        logger.info(f"🔍 Scan at {now.strftime('%H:%M:%S UTC')}")

        for coin_name, coin_ticker in COINS.items():
            for tf_label, tf_interval in TIMEFRAMES.items():

                comp_df = get_btcd(tf_label)
                if comp_df is None:
                    logger.warning(
                        f"No BTC.D cache yet for {tf_label} — "
                        f"skipping {coin_name}"
                    )
                    continue

                coin_df = get_ohlc_binance(coin_ticker, tf_interval)
                if coin_df is None:
                    continue

                for sig in detect_smt(coin_df, comp_df, COMP_LABEL):
                    sig_key = (
                        f"{coin_name}_{tf_label}_{sig['direction']}_"
                        f"{sig['coin_bar_a']}_{sig['coin_bar_b']}_"
                        f"{round(sig['coin_px_a'],4)}_{round(sig['coin_px_b'],4)}"
                    )
                    if sig_key in sent_signatures:
                        continue
                    ck = f"{coin_name}_{tf_label}_{sig['direction']}_BTCD"
                    if time.time() - last_alerts.get(ck, 0) < COOLDOWN_SECONDS:
                        continue

                    send_msg(format_signal(coin_name, tf_label, sig))
                    sent_signatures.add(sig_key)
                    last_alerts[ck] = time.time()
                    logger.info(
                        f"✅ {sig['direction']} SMT {coin_name} {tf_label} | "
                        f"coin={sig['coin_px_b']:.4f} "
                        f"btcd={sig['comp_px_b']:.3f}%"
                    )

                time.sleep(0.3)

        if len(sent_signatures) > 2000:
            for s in list(sent_signatures)[:1000]:
                sent_signatures.discard(s)

        logger.info(
            f"✅ Scan complete — {len(sent_signatures)} signals sent so far"
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
                f"latest = {df['close'].iloc[-1]:.3f}% | "
                f"age = {age}s</li>"
            )
        else:
            rows += f"<li><b>{tf}</b>: loading… (refreshes every 15 min)</li>"
    coins_html = ''.join(f"<li>{c}/USDT</li>" for c in COINS)
    return (
        f"<h2>🤖 SMT Alert Bot</h2>"
        f"<p><b>Status:</b> ✅ Running</p>"
        f"<p><b>Time UTC:</b> {now.strftime('%Y-%m-%d %H:%M:%S')}</p>"
        f"<p><b>Total alerts:</b> {len(sent_signatures)}</p>"
        f"<p><b>Cooldowns:</b> {len(last_alerts)}</p>"
        f"<h3>Monitoring:</h3><ul>{coins_html}</ul>"
        f"<h3>BTC.D Cache:</h3><ul>{rows}</ul>"
        f"<p><a href='/btcd'>View BTC.D candles</a> | "
        f"<a href='/scan_now'>Force scan</a> | "
        f"<a href='/refresh_btcd'>Force BTC.D refresh</a></p>"
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
        'btcd_cache':  info,
    }


@app.route('/btcd')
def show_btcd():
    out = []
    for tf in TIMEFRAMES:
        df = get_btcd(tf)
        if df is not None:
            out.append(
                f"<h3>BTC.D {tf} — last 10 candles</h3>"
                f"<pre>{df.tail(10).to_string(index=False)}</pre>"
            )
        else:
            out.append(f"<h3>BTC.D {tf}</h3><p>Not loaded yet.</p>")
    return ''.join(out) or '<p>No BTC.D data yet.</p>'


@app.route('/scan_now')
def scan_now_route():
    threading.Thread(target=scan_all, daemon=True).start()
    return 'Scan triggered!', 200


@app.route('/refresh_btcd')
def refresh_btcd_route():
    threading.Thread(target=refresh_btcd_cache, daemon=True).start()
    return 'BTC.D refresh triggered! Check /btcd in ~30 seconds.', 200


@app.route('/reset')
def reset():
    sent_signatures.clear()
    last_alerts.clear()
    with BTCD_LOCK:
        BTCD_CACHE.clear()
        BTCD_CACHE_TS.clear()
    return 'Reset done!', 200


# ============================================================
# STARTUP
# ============================================================

def start_bot():
    logger.info("SMT Bot starting — BTC.D mode")

    try:
        send_startup_msg()
    except Exception as e:
        logger.error(f"Startup msg error: {e}")

    # ── Initial BTC.D fetch (30s delay to let CoinGecko recover) ─
    def delayed_initial_fetch():
        logger.info("Waiting 30s before first CoinGecko fetch…")
        time.sleep(30)
        refresh_btcd_cache()

    threading.Thread(target=delayed_initial_fetch, daemon=True).start()

    # ── Scheduler ────────────────────────────────────────────
    scheduler = BackgroundScheduler(timezone='UTC')

    # Scan every minute
    scheduler.add_job(
        scan_all,
        trigger='cron', minute='*',
        id='scan_job',
        max_instances=1, coalesce=True, misfire_grace_time=30,
    )

    # Refresh BTC.D every 15 minutes (offset by 1 min so it finishes before scan)
    scheduler.add_job(
        refresh_btcd_cache,
        trigger='cron', minute='1,16,31,46',
        id='btcd_refresh_job',
        max_instances=1, coalesce=True, misfire_grace_time=60,
    )

    scheduler.start()
    logger.info(
        "Scheduler started — "
        "scan every min | BTC.D refresh at :01 :16 :31 :46"
    )


threading.Thread(target=start_bot, daemon=True).start()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
