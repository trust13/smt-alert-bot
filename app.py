# ============================================================
# ORDER BLOCK ALERT BOT — LIFECYCLE MODE (REAL-TIME ONLY)
# Coins: BTC, ETH, SOL  |  Timeframes: 15m, 1H
#
# Three alert states:
#   🟡 PRE-ALERT  → OB candidate detected (fractal JUST confirmed)
#   🟢 CONFIRMED  → candle closes beyond fractal in OB direction
#   ❌ FAILED     → candle closes outside OB zone (opposite direction)
#
# REAL-TIME ONLY: Pre-alerts only fire for fractals confirmed by the
# very last closed candle. Anything older is ignored.
#
# Data: Binance.US (no external APIs, no API keys)
# Scan: Every 1 minute
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

# ── Coins & Timeframes ───────────────────────────────────────
COINS = {
    'BTC': 'BTCUSDT',
    'ETH': 'ETHUSDT',
    'SOL': 'SOLUSDT',
}
TIMEFRAMES = {
    '15m': '15m',
    '1h':  '1h',
}

BINANCE_US_BASE  = 'https://api.binance.us/api/v3'
BINANCE_INTERVAL = {'15m': '15m', '1h': '1h'}

# ── OB Settings ──────────────────────────────────────────────
FRACTAL_TYPE        = 3
REQUIRE_FVG         = True
MAX_OB_TO_FVG_BARS  = 3
OB_SEARCH_LOOKBACK  = 5
SKIP_OVERLAP        = True
OVERLAP_WINDOW      = 20
FRESH_ONLY          = True

# ── REAL-TIME FRESHNESS ──────────────────────────────────────
# Only consider fractals whose confirming candle is the JUST-CLOSED bar.
# 1 = real-time only (fractal confirmed by the very last closed candle).
# Anything > 1 = looking at past fractals (NOT recommended).
MAX_PREALERT_AGE_BARS = 1

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
# PERSISTENT PENDING OB STATE
# ============================================================

pending_obs   = {}
pending_lock  = threading.Lock()
prealert_sent = set()


# ============================================================
# BINANCE.US DATA
# ============================================================

def fetch_klines(symbol, interval, limit=500):
    try:
        resp = requests.get(
            f"{BINANCE_US_BASE}/klines",
            params={'symbol': symbol, 'interval': interval, 'limit': limit},
            timeout=15,
        )
        if resp.status_code != 200:
            logger.error(f"Binance.US {resp.status_code} — {symbol}/{interval}")
            return None
        klines = resp.json()
        if not klines or isinstance(klines, dict):
            return None
        df = pd.DataFrame(klines, columns=[
            'time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'qav', 'num_trades',
            'taker_base', 'taker_quote', 'ignore',
        ])
        df['time']       = pd.to_datetime(df['time'],       unit='ms', utc=True)
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].astype(float)
        return df[['time', 'open', 'high', 'low', 'close', 'close_time']].reset_index(drop=True)
    except Exception as e:
        logger.error(f"fetch_klines {symbol}/{interval}: {e}")
        return None


# ============================================================
# FRACTAL DETECTION
# ============================================================

def is_fractal_high(highs, idx, side):
    n = len(highs)
    if idx - side < 0 or idx + side >= n: return False
    pivot = highs[idx]
    for k in range(1, side + 1):
        if highs[idx - k] >= pivot: return False
        if highs[idx + k] >= pivot: return False
    return True


def is_fractal_low(lows, idx, side):
    n = len(lows)
    if idx - side < 0 or idx + side >= n: return False
    pivot = lows[idx]
    for k in range(1, side + 1):
        if lows[idx - k] <= pivot: return False
        if lows[idx + k] <= pivot: return False
    return True


# ============================================================
# FVG DETECTION
# ============================================================

def has_bullish_fvg_after(highs, lows, ob_idx, max_bars, n):
    end = min(n - 2, ob_idx + max_bars)
    for i in range(ob_idx, end + 1):
        if i + 2 >= n: break
        if lows[i + 2] > highs[i]:
            return True
    return False


def has_bearish_fvg_after(highs, lows, ob_idx, max_bars, n):
    end = min(n - 2, ob_idx + max_bars)
    for i in range(ob_idx, end + 1):
        if i + 2 >= n: break
        if highs[i + 2] < lows[i]:
            return True
    return False


def is_bullish(o, c): return c > o
def is_bearish(o, c): return c < o


# ============================================================
# DETECT NEW PENDING OBs FROM JUST-CONFIRMED FRACTALS
# ============================================================

def detect_new_pending(df, coin, tf):
    """
    Look for fractals whose right-side neighbor is the JUST-CLOSED bar.
    For each, find the OB candidate. If valid, return as pending.
    """
    n = len(df)
    side = 1 if FRACTAL_TYPE == 3 else 2
    if n < (side * 2 + 10):
        return []

    opens  = df['open'].values
    highs  = df['high'].values
    lows   = df['low'].values
    closes = df['close'].values
    times  = df['time'].values

    last_closed_idx = n - 2

    # Real-time only: the fractal pivot must be exactly `side` bars
    # before the last closed bar, so its right neighbor IS the last closed bar.
    # MAX_PREALERT_AGE_BARS = 1 → check only the freshest possible fractal
    # MAX_PREALERT_AGE_BARS > 1 → also check older fractals (not recommended)

    new_pending = []

    # Range of fractal pivot indices to check
    # The freshest fractal pivot = last_closed_idx - side
    # We also accept up to MAX_PREALERT_AGE_BARS - 1 older ones
    newest_pivot_idx = last_closed_idx - side
    oldest_pivot_idx = newest_pivot_idx - (MAX_PREALERT_AGE_BARS - 1)
    oldest_pivot_idx = max(side, oldest_pivot_idx)

    for f_idx in range(oldest_pivot_idx, newest_pivot_idx + 1):
        if f_idx + side > last_closed_idx:
            continue

        # ── Bullish OB candidate (after fractal high) ─────
        if is_fractal_high(highs, f_idx, side):
            f_level = highs[f_idx]
            f_time  = times[f_idx]

            ob_idx = -1
            for k in range(f_idx - 1, max(0, f_idx - 1 - OB_SEARCH_LOOKBACK), -1):
                if is_bearish(opens[k], closes[k]):
                    ob_idx = k
                    break

            if ob_idx >= 0:
                fvg_ok = True
                if REQUIRE_FVG:
                    fvg_ok = has_bullish_fvg_after(
                        highs, lows, ob_idx, MAX_OB_TO_FVG_BARS, n
                    )

                ob_top = opens[ob_idx]
                ob_bot = lows[ob_idx]

                fresh = True
                if FRESH_ONLY:
                    for k in range(ob_idx + 1, last_closed_idx + 1):
                        if closes[k] < ob_bot:
                            fresh = False
                            break

                if fvg_ok and fresh:
                    key = f"{coin}_{tf}_BULL_{pd.Timestamp(f_time).isoformat()}"
                    new_pending.append({
                        'key':           key,
                        'coin':          coin,
                        'tf':            tf,
                        'type':          'BULL',
                        'fractal_level': f_level,
                        'fractal_time':  f_time,
                        'ob_top':        ob_top,
                        'ob_bot':        ob_bot,
                        'ob_time':       times[ob_idx],
                    })

        # ── Bearish OB candidate (after fractal low) ──────
        if is_fractal_low(lows, f_idx, side):
            f_level = lows[f_idx]
            f_time  = times[f_idx]

            ob_idx = -1
            for k in range(f_idx - 1, max(0, f_idx - 1 - OB_SEARCH_LOOKBACK), -1):
                if is_bullish(opens[k], closes[k]):
                    ob_idx = k
                    break

            if ob_idx >= 0:
                fvg_ok = True
                if REQUIRE_FVG:
                    fvg_ok = has_bearish_fvg_after(
                        highs, lows, ob_idx, MAX_OB_TO_FVG_BARS, n
                    )

                ob_top = highs[ob_idx]
                ob_bot = opens[ob_idx]

                fresh = True
                if FRESH_ONLY:
                    for k in range(ob_idx + 1, last_closed_idx + 1):
                        if closes[k] > ob_top:
                            fresh = False
                            break

                if fvg_ok and fresh:
                    key = f"{coin}_{tf}_BEAR_{pd.Timestamp(f_time).isoformat()}"
                    new_pending.append({
                        'key':           key,
                        'coin':          coin,
                        'tf':            tf,
                        'type':          'BEAR',
                        'fractal_level': f_level,
                        'fractal_time':  f_time,
                        'ob_top':        ob_top,
                        'ob_bot':        ob_bot,
                        'ob_time':       times[ob_idx],
                    })

    return new_pending


# ============================================================
# CHECK PENDING OBs FOR CONFIRMATION OR FAILURE
# Uses just-closed candle's CLOSE price
# ============================================================

def check_pending_status(coin, tf, df):
    if df is None or len(df) < 2:
        return [], []

    last_closed_close = float(df['close'].iloc[-2])
    last_closed_time  = df['time'].iloc[-2]

    confirmed = []
    failed    = []

    with pending_lock:
        keys_to_check = [
            k for k in pending_obs
            if pending_obs[k]['coin'] == coin
            and pending_obs[k]['tf'] == tf
        ]

        for key in keys_to_check:
            ob = pending_obs[key]

            if ob['fractal_time'] >= last_closed_time:
                continue

            if ob['type'] == 'BULL':
                if last_closed_close > ob['fractal_level']:
                    ob['confirm_close'] = last_closed_close
                    ob['confirm_time']  = last_closed_time
                    confirmed.append(ob)
                    del pending_obs[key]
                elif last_closed_close < ob['ob_bot']:
                    ob['fail_close'] = last_closed_close
                    ob['fail_time']  = last_closed_time
                    failed.append(ob)
                    del pending_obs[key]
            else:  # BEAR
                if last_closed_close < ob['fractal_level']:
                    ob['confirm_close'] = last_closed_close
                    ob['confirm_time']  = last_closed_time
                    confirmed.append(ob)
                    del pending_obs[key]
                elif last_closed_close > ob['ob_top']:
                    ob['fail_close'] = last_closed_close
                    ob['fail_time']  = last_closed_time
                    failed.append(ob)
                    del pending_obs[key]

    return confirmed, failed


# ============================================================
# TELEGRAM
# ============================================================

def send_msg(text):
    try:
        if bot is None or not CHAT_ID:
            return
        bot.send_message(CHAT_ID, text, parse_mode='HTML')
    except Exception as e:
        logger.error(f"Telegram send error: {e}")


def fmt_pre(ob):
    is_bull = ob['type'] == 'BULL'
    label   = 'POTENTIAL Bull OB' if is_bull else 'POTENTIAL Bear OB'
    return (
        f"🟡 PRE-ALERT — {ob['coin']} {ob['tf']} | {label}\n"
        f"Zone: {ob['ob_bot']:,.4f} – {ob['ob_top']:,.4f}\n"
        f"Watch: {ob['fractal_level']:,.4f}"
    )


def fmt_confirmed(ob):
    is_bull = ob['type'] == 'BULL'
    emoji   = '🟢' if is_bull else '🔴'
    label   = 'Bull OB' if is_bull else 'Bear OB'
    return (
        f"{emoji} CONFIRMED — {ob['coin']} {ob['tf']} | {label}\n"
        f"Zone: {ob['ob_bot']:,.4f} – {ob['ob_top']:,.4f}"
    )


def fmt_failed(ob):
    is_bull = ob['type'] == 'BULL'
    label   = 'Bull OB' if is_bull else 'Bear OB'
    return (
        f"❌ FAILED — {ob['coin']} {ob['tf']} | {label}\n"
        f"Zone was: {ob['ob_bot']:,.4f} – {ob['ob_top']:,.4f}\n"
        f"Closed at: {ob['fail_close']:,.4f}"
    )


def send_startup_msg():
    age_label = (
        "real-time only (just-confirmed fractals)"
        if MAX_PREALERT_AGE_BARS == 1
        else f"up to {MAX_PREALERT_AGE_BARS} bars old"
    )
    send_msg(
        f"🤖 <b>OB Alert Bot — REAL-TIME LIFECYCLE</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Coins:</b> {', '.join(COINS)}\n"
        f"<b>Timeframes:</b> {', '.join(TIMEFRAMES)}\n\n"
        f"<b>3 Alert States:</b>\n"
        f"  🟡 PRE-ALERT — OB candidate detected (fresh fractal)\n"
        f"  🟢 CONFIRMED — candle closes beyond fractal\n"
        f"  ❌ FAILED — candle closes outside OB zone\n\n"
        f"<b>Settings:</b>\n"
        f"  Fractal: {FRACTAL_TYPE}-bar\n"
        f"  FVG required: {REQUIRE_FVG}\n"
        f"  OB lookback: {OB_SEARCH_LOOKBACK}\n"
        f"  Skip overlap: {SKIP_OVERLAP}\n"
        f"  Fresh only: {FRESH_ONLY}\n"
        f"  Pre-alert mode: {age_label}\n\n"
        f"<b>Scan:</b> Every 1 minute\n\n"
        f"<i>Data: Binance.US</i>"
    )


# ============================================================
# SCAN LOOP
# ============================================================

def scan_all():
    try:
        now = datetime.now(timezone.utc)
        logger.info(f"🔍 OB scan at {now.strftime('%H:%M:%S UTC')}")

        for coin_name, coin_ticker in COINS.items():
            for tf_label, tf_interval in TIMEFRAMES.items():
                df = fetch_klines(
                    coin_ticker,
                    BINANCE_INTERVAL[tf_interval],
                    limit=500,
                )
                if df is None or len(df) < 50:
                    logger.warning(f"No data: {coin_name} {tf_label}")
                    continue

                # ── Step 1: Check pending OBs ─────────────
                confirmed, failed = check_pending_status(coin_name, tf_label, df)

                for ob in confirmed:
                    send_msg(fmt_confirmed(ob))
                    direction = '>' if ob['type'] == 'BULL' else '<'
                    logger.info(
                        f"✅ CONFIRMED {ob['type']} {coin_name} {tf_label} | "
                        f"close={ob['confirm_close']:.4f} {direction} "
                        f"fract={ob['fractal_level']:.4f}"
                    )

                for ob in failed:
                    send_msg(fmt_failed(ob))
                    logger.info(
                        f"❌ FAILED {ob['type']} {coin_name} {tf_label} | "
                        f"close={ob['fail_close']:.4f}"
                    )

                # ── Step 2: Detect new pending OBs ──────────
                new_obs = detect_new_pending(df, coin_name, tf_label)
                with pending_lock:
                    for ob in new_obs:
                        if ob['key'] in pending_obs:
                            continue
                        if ob['key'] in prealert_sent:
                            continue
                        pending_obs[ob['key']] = ob
                        prealert_sent.add(ob['key'])
                        send_msg(fmt_pre(ob))
                        logger.info(
                            f"🟡 PRE-ALERT {ob['type']} {coin_name} {tf_label} | "
                            f"zone[{ob['ob_bot']:.4f}–{ob['ob_top']:.4f}] | "
                            f"watch={ob['fractal_level']:.4f}"
                        )

                time.sleep(0.2)

        if len(prealert_sent) > 5000:
            for s in list(prealert_sent)[:2500]:
                prealert_sent.discard(s)

        with pending_lock:
            pend_count = len(pending_obs)

        logger.info(
            f"✅ Scan complete — {pend_count} pending | "
            f"{len(prealert_sent)} pre-alerts (lifetime)"
        )

    except Exception as e:
        logger.error(f"scan_all error: {e}")


# ============================================================
# FLASK ROUTES
# ============================================================

@app.route('/')
def index():
    now = datetime.now(timezone.utc)
    coins_html = ''.join(f"<li>{c}/USDT</li>" for c in COINS)
    tfs_html   = ''.join(f"<li>{tf}</li>" for tf in TIMEFRAMES)

    with pending_lock:
        pending_html = ''.join(
            f"<li>{ob['coin']} {ob['tf']} {ob['type']} | "
            f"zone {ob['ob_bot']:.4f}-{ob['ob_top']:.4f} | "
            f"watch {ob['fractal_level']:.4f}</li>"
            for ob in pending_obs.values()
        ) or '<li>(none)</li>'

    return (
        f"<h2>🤖 Order Block Alert Bot — Real-Time Lifecycle</h2>"
        f"<p><b>Status:</b> ✅ Running</p>"
        f"<p><b>Time UTC:</b> {now.strftime('%Y-%m-%d %H:%M:%S')}</p>"
        f"<p><b>Mode:</b> 3-state lifecycle (pre / confirmed / failed)</p>"
        f"<p><b>Pre-alert age limit:</b> {MAX_PREALERT_AGE_BARS} bar "
        f"({'real-time only' if MAX_PREALERT_AGE_BARS == 1 else 'historical lookback'})</p>"
        f"<p><b>Data:</b> Binance.US | <b>Scan:</b> 1 min</p>"
        f"<p><b>Pre-alerts sent (lifetime):</b> {len(prealert_sent)}</p>"
        f"<h3>Currently Pending OBs:</h3><ul>{pending_html}</ul>"
        f"<h3>Monitoring:</h3><ul>{coins_html}</ul>"
        f"<h3>Timeframes:</h3><ul>{tfs_html}</ul>"
        f"<p>"
        f"<a href='/scan_now'>Force scan</a> | "
        f"<a href='/status'>JSON status</a> | "
        f"<a href='/reset'>Reset state</a>"
        f"</p>"
    )


@app.route('/health')
def health():
    return 'OK', 200


@app.route('/status')
def status():
    with pending_lock:
        pend_list = [
            {
                'coin': ob['coin'],
                'tf':   ob['tf'],
                'type': ob['type'],
                'ob_top': ob['ob_top'],
                'ob_bot': ob['ob_bot'],
                'fractal_level': ob['fractal_level'],
            }
            for ob in pending_obs.values()
        ]
    return {
        'status':         'running',
        'mode':           'Real-time lifecycle',
        'time_utc':       datetime.now(timezone.utc).isoformat(),
        'coins':          list(COINS.keys()),
        'timeframes':     list(TIMEFRAMES.keys()),
        'pending_count':  len(pend_list),
        'pending_obs':    pend_list,
        'prealerts_sent_lifetime': len(prealert_sent),
        'settings': {
            'fractal_type':           FRACTAL_TYPE,
            'require_fvg':            REQUIRE_FVG,
            'max_ob_to_fvg_bars':     MAX_OB_TO_FVG_BARS,
            'ob_search_lookback':     OB_SEARCH_LOOKBACK,
            'skip_overlap':           SKIP_OVERLAP,
            'overlap_window':         OVERLAP_WINDOW,
            'fresh_only':             FRESH_ONLY,
            'max_prealert_age_bars':  MAX_PREALERT_AGE_BARS,
        },
    }


@app.route('/scan_now')
def scan_now_route():
    threading.Thread(target=scan_all, daemon=True).start()
    return 'Scan triggered!', 200


@app.route('/reset')
def reset():
    global pending_obs, prealert_sent
    with pending_lock:
        pending_obs.clear()
    prealert_sent.clear()
    return 'Reset done!', 200


# ============================================================
# STARTUP
# ============================================================

def start_bot():
    logger.info("Order Block Bot starting (Real-Time Lifecycle mode)…")
    try:
        send_startup_msg()
    except Exception as e:
        logger.error(f"Startup msg error: {e}")

    scheduler = BackgroundScheduler(timezone='UTC')
    scheduler.add_job(
        scan_all,
        trigger='cron',
        minute='*',
        id='ob_scan_job',
        max_instances=1,
        coalesce=True,
        misfire_grace_time=20,
    )
    scheduler.start()
    logger.info("Scheduler started — OB scan every 1 minute")

    threading.Thread(target=scan_all, daemon=True).start()


threading.Thread(target=start_bot, daemon=True).start()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
