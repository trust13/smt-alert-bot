# ============================================================
# SMT ALERT BOT — TraderDiegoX Logic Mirror
# Coins: BTC, ETH, SOL, BNB
# Timeframes: 15m, 1H
# Comparison: BTC Dominance (BTC.D) via CoinGecko
# Crypto data: Binance.US public API
# ============================================================

import os
import time
import logging
import threading
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
import telebot

# ── Config ───────────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '')
CHAT_ID        = os.environ.get('CHAT_ID',        '')

# ── Coins ────────────────────────────────────────────────────
COINS = {
    'BTC': 'BTCUSDT',
    'ETH': 'ETHUSDT',
    'SOL': 'SOLUSDT',
    'BNB': 'BNBUSDT',
}

COMP_LABEL = 'BTC.D'

# ── Timeframes ───────────────────────────────────────────────
TIMEFRAMES = {
    '15m': '15m',
    '1h':  '1h',
}

# ── API Endpoints ────────────────────────────────────────────
BINANCE_US_BASE = 'https://api.binance.us/api/v3'
COINGECKO_BASE  = 'https://api.coingecko.com/api/v3'

BINANCE_INTERVAL = {
    '15m': '15m',
    '1h':  '1h',
}

# ── SMT Settings ─────────────────────────────────────────────
PIVOT_LOOKBACK   = 1
PIVOT_A_STRENGTH = 2
SYNC_TOL         = 2
MAX_SIGNAL_SPAN  = 180
CROSS_TOL_PCT    = 0.02

# ── Cooldown ─────────────────────────────────────────────────
COOLDOWN_SECONDS = 30 * 60

# ── BTC.D Cache — ONE shared cache, refreshed every 5 minutes ─
# Fetch BTC.D once per 5 min regardless of how many coins scan
BTCD_CACHE: dict[str, pd.DataFrame] = {}   # tf -> DataFrame
BTCD_CACHE_TS: dict[str, float]     = {}   # tf -> epoch seconds
BTCD_CACHE_TTL = 5 * 60                    # 5 minutes

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
# BTC DOMINANCE — fetched ONCE per timeframe per 5 min
# ============================================================

def _fetch_btcd_fresh(interval: str) -> pd.DataFrame | None:
    """
    Make the actual HTTP calls to CoinGecko.
    Called only when cache is stale — at most once per 5 minutes.
    Two API calls total per timeframe refresh.
    """
    days = 2 if interval == '15m' else 90
    resample_rule = '15min' if interval == '15m' else '1h'

    try:
        # ── Call 1: BTC market cap ───────────────────────────
        r1 = requests.get(
            f"{COINGECKO_BASE}/coins/bitcoin/market_chart",
            params={
                'vs_currency': 'usd',
                'days':        days,
                'interval':    'hourly' if interval == '1h' else 'minutely',
            },
            timeout=25,
            headers={'Accept': 'application/json'},
        )

        if r1.status_code == 429:
            logger.warning(f"CoinGecko 429 on BTC market_chart ({interval})")
            return None
        if r1.status_code != 200:
            logger.error(
                f"CoinGecko BTC market_chart HTTP {r1.status_code} ({interval})"
            )
            return None

        btc_mcaps = r1.json().get('market_caps', [])
        if not btc_mcaps:
            logger.error("CoinGecko BTC market_caps is empty")
            return None

        btc_df = (
            pd.DataFrame(btc_mcaps, columns=['ts_ms', 'btc_mcap'])
            .assign(time=lambda d: pd.to_datetime(d['ts_ms'], unit='ms', utc=True))
            [['time', 'btc_mcap']]
            .set_index('time')
        )

        # ── Polite pause between calls ───────────────────────
        time.sleep(2)

        # ── Call 2: Total crypto market cap ─────────────────
        r2 = requests.get(
            f"{COINGECKO_BASE}/global/market_cap_chart",
            params={'days': days},
            timeout=25,
            headers={'Accept': 'application/json'},
        )

        if r2.status_code == 429:
            logger.warning(f"CoinGecko 429 on total market_cap_chart ({interval})")
            return None
        if r2.status_code != 200:
            logger.error(
                f"CoinGecko total market_cap_chart HTTP {r2.status_code} ({interval})"
            )
            return None

        raw_total = r2.json().get('market_cap_chart', {}).get('market_cap', [])
        if not raw_total:
            logger.error("CoinGecko total market_cap is empty")
            return None

        total_df = (
            pd.DataFrame(raw_total, columns=['ts_ms', 'total_mcap'])
            .assign(time=lambda d: pd.to_datetime(d['ts_ms'], unit='ms', utc=True))
            [['time', 'total_mcap']]
            .set_index('time')
        )

        # ── Merge & compute BTC.D ────────────────────────────
        merged = (
            btc_df
            .join(total_df, how='outer')
            .sort_index()
            .ffill()
            .dropna()
        )
        if len(merged) < 20:
            logger.error(f"BTC.D merged too short: {len(merged)} rows")
            return None

        merged['btcd'] = (merged['btc_mcap'] / merged['total_mcap']) * 100.0

        # ── Resample into OHLC candles ───────────────────────
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
            logger.error(f"BTC.D OHLC too short after resample: {len(ohlc)}")
            return None

        logger.info(
            f"✅ BTC.D {interval}: {len(ohlc)} candles | "
            f"latest = {ohlc['close'].iloc[-1]:.3f}%"
        )
        return ohlc

    except Exception as e:
        logger.error(f"_fetch_btcd_fresh {interval}: {e}")
        return None


def get_btcd_ohlc(interval: str) -> pd.DataFrame | None:
    """
    Return BTC.D OHLC for the given timeframe.
    Uses a 5-minute cache — so no matter how many coins scan,
    CoinGecko is called at most once per 5 minutes per timeframe.
    """
    now = time.time()
    last_fetch = BTCD_CACHE_TS.get(interval, 0)

    if now - last_fetch < BTCD_CACHE_TTL and interval in BTCD_CACHE:
        return BTCD_CACHE[interval]          # serve from cache

    logger.info(f"Refreshing BTC.D cache for {interval}…")
    fresh = _fetch_btcd_fresh(interval)

    if fresh is not None:
        BTCD_CACHE[interval]    = fresh
        BTCD_CACHE_TS[interval] = now
        return fresh

    # If fetch failed, keep serving stale cache rather than None
    if interval in BTCD_CACHE:
        logger.warning(f"Using stale BTC.D cache for {interval}")
        return BTCD_CACHE[interval]

    return None


# ============================================================
# BINANCE.US — crypto OHLC
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
    if span <= 0:            return True
    if span > MAX_SIGNAL_SPAN: return False
    for d in range(1, span + 1):
        idx = bar_a + d
        if idx >= len(highs): return False
        line_px = px_a + (px_b - px_a) * d / (bar_b - bar_a)
        tol_px  = abs(line_px) * CROSS_TOL_PCT / 100.0
        if is_bear and highs[idx] > line_px + tol_px: return False
        if not is_bear and lows[idx] < line_px - tol_px: return False
    return True


def near_pivot(pivots, target_bar: int, tol: int) -> int:
    best, best_d = -1, tol + 1
    for i, (b, _) in enumerate(pivots):
        d = abs(b - target_bar)
        if d <= tol and d < best_d:
            best_d, best = d, i
    return best


def near_pivot_before(pivots, target_bar: int, limit_bar: int, tol: int) -> int:
    best, best_d = -1, tol + 1
    for i, (b, _) in enumerate(pivots):
        if b >= limit_bar: continue
        d = abs(b - target_bar)
        if d <= tol and d < best_d:
            best_d, best = d, i
    return best


# ============================================================
# SMT DETECTION
# ============================================================

def detect_smt(coin_df: pd.DataFrame,
               comp_df: pd.DataFrame,
               comp_label: str) -> list:
    if coin_df is None or comp_df is None:       return []
    if len(coin_df) < 50 or len(comp_df) < 50:  return []

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

    # ── BEAR: coin Higher High + comp Lower High ─────────────
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
                                     bar_a,  px_a,  bar_b,  px_b,  True) and
                            cross_ok(xv['high'], xv['low'],
                                     xbar_a, xpx_a, xbar_b, xpx_b, True)):
                            signals.append({
                                'direction':  'BEAR',
                                'comp_label': comp_label,
                                'coin_bar_a': bar_a,  'coin_px_a': px_a,
                                'coin_bar_b': bar_b,  'coin_px_b': px_b,
                                'comp_bar_a': xbar_a, 'comp_px_a': xpx_a,
                                'comp_bar_b': xbar_b, 'comp_px_b': xpx_b,
                                'span':  span,
                                'time_b': coin_df['time'].iloc[bar_b],
                            })
                            break

    # ── BULL: coin Lower Low + comp Higher Low ───────────────
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
                                     bar_a,  px_a,  bar_b,  px_b,  False) and
                            cross_ok(xv['high'], xv['low'],
                                     xbar_a, xpx_a, xbar_b, xpx_b, False)):
                            signals.append({
                                'direction':  'BULL',
                                'comp_label': comp_label,
                                'coin_bar_a': bar_a,  'coin_px_a': px_a,
                                'coin_bar_b': bar_b,  'coin_px_b': px_b,
                                'comp_bar_a': xbar_a, 'comp_px_a': xpx_a,
                                'comp_bar_b': xbar_b, 'comp_px_b': xpx_b,
                                'span':  span,
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
        arrow_coin = '↓ Lower Low'
        arrow_comp = '↑ Higher Low  ← BTC.D DIVERGED'
        watch      = '🎯 Watch for reversal UP'
        note       = 'BTC Dominance rising while coin drops → altcoin underperforming → reversal UP expected'
    else:
        arrow_coin = '↑ Higher High'
        arrow_comp = '↓ Lower High  ← BTC.D DIVERGED'
        watch      = '🎯 Watch for reversal DOWN'
        note       = 'BTC Dominance falling while coin pumps → altcoin overperforming → reversal DOWN expected'

    ts = pd.Timestamp(sig['time_b']).strftime('%H:%M UTC %d-%b')

    return (
        f"{emoji} <b>{d} SMT — {coin}/USDT {tf}</b>\n"
        f"{'─'*32}\n"
        f"<b>Coin Pivot A:</b>   ${sig['coin_px_a']:,.4f}  ({sig['span']} bars ago)\n"
        f"<b>Coin Pivot B:</b>   ${sig['coin_px_b']:,.4f}  (now)\n"
        f"<b>Coin Move:</b>      {arrow_coin}\n\n"
        f"<b>BTC.D Pivot A:</b>  {sig['comp_px_a']:.3f}%\n"
        f"<b>BTC.D Pivot B:</b>  {sig['comp_px_b']:.3f}%\n"
        f"<b>BTC.D Move:</b>     {arrow_comp}\n\n"
        f"<b>Span:</b>  {sig['span']} bars\n"
        f"<b>Time:</b>  {ts}\n\n"
        f"<i>{note}</i>\n\n"
        f"{watch}"
    )


def send_startup_msg():
    send_msg(
        f"🤖 <b>SMT Alert Bot — LIVE</b>\n"
        f"{'─'*32}\n"
        f"<b>Coins:</b>       {', '.join(COINS)}\n"
        f"<b>Timeframes:</b>  {', '.join(TIMEFRAMES)}\n"
        f"<b>Comparison:</b>  BTC Dominance (BTC.D)\n\n"
        f"<b>Data Sources:</b>\n"
        f"  Crypto:  Binance.US API  (live, no key)\n"
        f"  BTC.D:   CoinGecko API   (live, no key)\n\n"
        f"<b>SMT Logic:</b>\n"
        f"  🟢 BULL = Coin Lower Low  + BTC.D Higher Low\n"
        f"  🔴 BEAR = Coin Higher High + BTC.D Lower High\n\n"
        f"<b>BTC.D cache TTL:</b>  5 minutes\n"
        f"<b>Alert cooldown:</b>   30 minutes\n\n"
        f"<i>Scanning every minute…</i>"
    )


# ============================================================
# CORE SCAN
# ============================================================

def scan_all():
    try:
        now = datetime.now(timezone.utc)
        logger.info(f"🔍 Scan cycle at {now.strftime('%H:%M:%S UTC')}")

        # ── Fetch BTC.D ONCE per timeframe for the whole cycle ──
        btcd: dict[str, pd.DataFrame | None] = {}
        for tf_label in TIMEFRAMES:
            btcd[tf_label] = get_btcd_ohlc(tf_label)
            # If we just fetched fresh data, wait a moment
            # before the next timeframe fetch
            if BTCD_CACHE_TS.get(tf_label, 0) >= now.timestamp() - 5:
                time.sleep(3)

        # ── Scan every coin against the pre-fetched BTC.D ───────
        for coin_name, coin_ticker in COINS.items():
            for tf_label, tf_interval in TIMEFRAMES.items():

                comp_df = btcd.get(tf_label)
                if comp_df is None:
                    logger.warning(
                        f"Skipping {coin_name} {tf_label} — no BTC.D data"
                    )
                    continue

                coin_df = get_ohlc_binance(coin_ticker, tf_interval)
                if coin_df is None:
                    logger.warning(f"No Binance data: {coin_name} {tf_label}")
                    continue

                signals = detect_smt(coin_df, comp_df, COMP_LABEL)

                for sig in signals:
                    signature = (
                        f"{coin_name}_{tf_label}_{sig['direction']}_BTCD_"
                        f"{sig['coin_bar_a']}_{sig['coin_bar_b']}_"
                        f"{round(sig['coin_px_a'], 4)}_{round(sig['coin_px_b'], 4)}"
                    )
                    if signature in sent_signatures:
                        continue

                    ck = f"{coin_name}_{tf_label}_{sig['direction']}_BTCD"
                    if time.time() - last_alerts.get(ck, 0) < COOLDOWN_SECONDS:
                        continue

                    send_msg(format_signal(coin_name, tf_label, sig))
                    sent_signatures.add(signature)
                    last_alerts[ck] = time.time()
                    logger.info(
                        f"✅ {sig['direction']} SMT {coin_name} {tf_label} | "
                        f"coin={sig['coin_px_b']:.4f} btcd={sig['comp_px_b']:.3f}%"
                    )

                time.sleep(0.3)   # gentle pace between Binance calls

        # Trim signature set
        if len(sent_signatures) > 2000:
            for s in list(sent_signatures)[:1000]:
                sent_signatures.discard(s)

        logger.info(
            f"✅ Scan complete — {len(sent_signatures)} total signals sent"
        )

    except Exception as e:
        logger.error(f"scan_all error: {e}")


# ============================================================
# FLASK ROUTES
# ============================================================

@app.route('/')
def index():
    now        = datetime.now(timezone.utc)
    coins_html = ''.join(f"<li>{c}/USDT</li>" for c in COINS)
    btcd_rows  = ''
    for tf in TIMEFRAMES:
        df  = BTCD_CACHE.get(tf)
        age = int(time.time() - BTCD_CACHE_TS.get(tf, 0))
        if df is not None:
            btcd_rows += (
                f"<li>{tf}: {len(df)} candles | "
                f"latest = {df['close'].iloc[-1]:.3f}% | "
                f"age = {age}s</li>"
            )
        else:
            btcd_rows += f"<li>{tf}: not yet loaded</li>"

    return (
        f"<h2>🤖 SMT Alert Bot</h2>"
        f"<p><b>Status:</b> ✅ Running</p>"
        f"<p><b>Time UTC:</b> {now.strftime('%Y-%m-%d %H:%M:%S')}</p>"
        f"<p><b>Comparison:</b> BTC Dominance (BTC.D)</p>"
        f"<p><b>Total alerts sent:</b> {len(sent_signatures)}</p>"
        f"<p><b>Active cooldowns:</b> {len(last_alerts)}</p>"
        f"<h3>Monitoring:</h3><ul>{coins_html}</ul>"
        f"<h3>BTC.D Cache:</h3><ul>{btcd_rows}</ul>"
    )


@app.route('/health')
def health():
    return 'OK', 200


@app.route('/status')
def status():
    btcd_info = {}
    for tf in TIMEFRAMES:
        df = BTCD_CACHE.get(tf)
        btcd_info[tf] = {
            'candles': len(df) if df is not None else 0,
            'latest':  float(df['close'].iloc[-1]) if df is not None else None,
            'age_sec': int(time.time() - BTCD_CACHE_TS.get(tf, 0)),
        }
    return {
        'status':      'running',
        'time_utc':    datetime.now(timezone.utc).isoformat(),
        'coins':       list(COINS.keys()),
        'timeframes':  list(TIMEFRAMES.keys()),
        'comparison':  'BTC.D',
        'alerts_sent': len(sent_signatures),
        'cooldowns':   len(last_alerts),
        'btcd_cache':  btcd_info,
    }


@app.route('/scan_now')
def scan_now_route():
    threading.Thread(target=scan_all, daemon=True).start()
    return 'Scan triggered!', 200


@app.route('/reset')
def reset():
    sent_signatures.clear()
    last_alerts.clear()
    BTCD_CACHE.clear()
    BTCD_CACHE_TS.clear()
    return 'Reset done — cache cleared!', 200


@app.route('/btcd')
def show_btcd():
    """Debug: show last 10 BTC.D candles per timeframe."""
    out = []
    for tf in TIMEFRAMES:
        df = BTCD_CACHE.get(tf)
        if df is not None:
            out.append(
                f"<h3>BTC.D {tf} — last 10 candles</h3>"
                f"<pre>{df.tail(10).to_string(index=False)}</pre>"
            )
        else:
            out.append(f"<h3>BTC.D {tf}</h3><p>No data in cache yet.</p>")
    return ''.join(out)


# ============================================================
# STARTUP
# ============================================================

def start_bot():
    logger.info("SMT Bot starting — BTC Dominance mode")

    try:
        send_startup_msg()
    except Exception as e:
        logger.error(f"Startup msg error: {e}")

    # Pre-warm cache — fetch 15m first, pause, then 1h
    logger.info("Pre-warming BTC.D cache…")
    for tf in ['15m', '1h']:
        df = get_btcd_ohlc(tf)
        if df is not None:
            logger.info(f"BTC.D {tf}: {len(df)} candles pre-loaded")
        else:
            logger.warning(f"BTC.D {tf}: pre-load failed (will retry on first scan)")
        time.sleep(4)   # 4s gap between the two fetches

    scheduler = BackgroundScheduler(timezone='UTC')
    scheduler.add_job(
        scan_all,
        trigger='cron',
        minute='*',
        id='scan_job',
        max_instances=1,
        coalesce=True,
        misfire_grace_time=30,
    )
    scheduler.start()
    logger.info("Scheduler started — scanning every minute")

    # Run first scan immediately
    threading.Thread(target=scan_all, daemon=True).start()


threading.Thread(target=start_bot, daemon=True).start()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
