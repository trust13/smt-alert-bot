# ============================================================
# ORDER BLOCK ALERT BOT — STRICT MODE + LATE PRE-ALERTS
# Coins: BTC, ETH, SOL  |  Timeframes: 15m, 1H
# Pre-alerts only fire LATE in candle (15m: min 8+, 1H: min 40+)
# Confirmed alerts only for OBs within last 3 candles
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

# ── Coins ────────────────────────────────────────────────────
COINS = {
    'BTC': 'BTCUSDT',
    'ETH': 'ETHUSDT',
    'SOL': 'SOLUSDT',
}

# ── Timeframes ───────────────────────────────────────────────
TIMEFRAMES = {
    '15m': '15m',
    '1h':  '1h',
}

# ── API ──────────────────────────────────────────────────────
BINANCE_US_BASE  = 'https://api.binance.us/api/v3'
BINANCE_INTERVAL = {'15m': '15m', '1h': '1h'}

# ── OB Detection Settings ────────────────────────────────────
FRACTAL_TYPE        = 3
REQUIRE_FVG         = True
MAX_OB_TO_FVG_BARS  = 3
OB_SEARCH_LOOKBACK  = 5
SKIP_OVERLAP        = True
OVERLAP_WINDOW      = 20
FRESH_ONLY          = True

# ── Strict Freshness ─────────────────────────────────────────
MAX_BREAK_AGE_BARS  = 3

# ── Pre-Alert Timing ─────────────────────────────────────────
# Only fire pre-alerts after the candle has been forming for
# at least this many minutes. Prevents early-candle false signals.
PREALERT_MIN_AGE_MINUTES = {
    '15m': 8,    # Pre-alert allowed from minute 8 onwards (8-15 in)
    '1h':  40,   # Pre-alert allowed from minute 40 onwards (40-60 in)
}

# ── Cooldown ─────────────────────────────────────────────────
COOLDOWN_SECONDS    = 30 * 60
MAX_OBS_PER_COIN    = 3

# ── Alert state ──────────────────────────────────────────────
last_alerts          = {}
sent_signatures      = set()
prealert_signatures  = set()

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
# BINANCE.US DATA
# ============================================================

def fetch_klines(symbol: str, interval: str,
                 limit: int = 500) -> pd.DataFrame | None:
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
# CANDLE AGE HELPER
# ============================================================

def is_candle_old_enough(df: pd.DataFrame, tf_label: str) -> tuple[bool, int]:
    """
    Returns (is_old_enough, current_age_minutes).
    Pre-alerts only fire after the live candle has aged enough.
    """
    if len(df) < 1:
        return False, 0
    live_open_time = df['time'].iloc[-1]
    now_utc = datetime.now(timezone.utc)
    age_seconds = (now_utc - live_open_time.to_pydatetime()).total_seconds()
    age_minutes = int(age_seconds // 60)
    min_age = PREALERT_MIN_AGE_MINUTES.get(tf_label, 0)
    return (age_minutes >= min_age, age_minutes)


# ============================================================
# FRACTAL DETECTION
# ============================================================

def is_fractal_high(highs, idx: int, side: int) -> bool:
    n = len(highs)
    if idx - side < 0 or idx + side >= n: return False
    pivot = highs[idx]
    for k in range(1, side + 1):
        if highs[idx - k] >= pivot: return False
        if highs[idx + k] >= pivot: return False
    return True


def is_fractal_low(lows, idx: int, side: int) -> bool:
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

def has_bullish_fvg_after(highs, lows, ob_idx, max_bars, n) -> int:
    end = min(n - 2, ob_idx + max_bars)
    for i in range(ob_idx, end + 1):
        if i + 2 >= n: break
        if lows[i + 2] > highs[i]:
            return i + 1
    return -1


def has_bearish_fvg_after(highs, lows, ob_idx, max_bars, n) -> int:
    end = min(n - 2, ob_idx + max_bars)
    for i in range(ob_idx, end + 1):
        if i + 2 >= n: break
        if highs[i + 2] < lows[i]:
            return i + 1
    return -1


# ============================================================
# OB DETECTION
# ============================================================

def is_bullish(o, c): return c > o
def is_bearish(o, c): return c < o


def detect_obs(df: pd.DataFrame) -> tuple[list, list]:
    """
    Returns (confirmed_obs, prealert_obs)
    Process bars chronologically like Pine Script.
    """
    n = len(df)
    side = 1 if FRACTAL_TYPE == 3 else 2
    if n < (side * 2 + 10):
        return [], []

    opens  = df['open'].values
    highs  = df['high'].values
    lows   = df['low'].values
    closes = df['close'].values
    times  = df['time'].values

    live_idx        = n - 1
    last_closed_idx = n - 2

    confirmed_obs = []
    prealert_obs  = []
    placed_bull_obs = []
    placed_bear_obs = []

    def overlaps_bull(top, bot, current_bar):
        if not SKIP_OVERLAP: return False
        for (t, b, idx) in placed_bull_obs:
            if current_bar - idx > OVERLAP_WINDOW: continue
            if not (bot > t or top < b):
                return True
        return False

    def overlaps_bear(top, bot, current_bar):
        if not SKIP_OVERLAP: return False
        for (t, b, idx) in placed_bear_obs:
            if current_bar - idx > OVERLAP_WINDOW: continue
            if not (bot > t or top < b):
                return True
        return False

    def is_bull_fresh(ob_bar, top, bot, until_bar):
        if not FRESH_ONLY: return True
        for k in range(ob_bar + 1, until_bar):
            if closes[k] < bot:
                return False
        return True

    def is_bear_fresh(ob_bar, top, bot, until_bar):
        if not FRESH_ONLY: return True
        for k in range(ob_bar + 1, until_bar):
            if closes[k] > top:
                return False
        return True

    active_fractal_high = None
    active_fractal_low  = None

    for i in range(side, last_closed_idx + 1):
        if i + side > last_closed_idx:
            break

        if is_fractal_high(highs, i, side):
            active_fractal_high = (i, highs[i])

        if is_fractal_low(lows, i, side):
            active_fractal_low = (i, lows[i])

        check_bar = i + side
        if check_bar > last_closed_idx:
            continue

        # BULLISH BREAK
        if active_fractal_high is not None:
            f_idx, f_level = active_fractal_high
            if check_bar > f_idx + side and closes[check_bar] > f_level:
                ob_idx = -1
                search_start = max(0, check_bar - OB_SEARCH_LOOKBACK)
                for k in range(check_bar - 1, search_start - 1, -1):
                    if is_bearish(opens[k], closes[k]):
                        ob_idx = k
                        break

                if ob_idx >= 0:
                    fvg_ok = True
                    if REQUIRE_FVG:
                        fvg_ok = has_bullish_fvg_after(
                            highs, lows, ob_idx, MAX_OB_TO_FVG_BARS, n
                        ) >= 0

                    top_lvl = opens[ob_idx]
                    bot_lvl = lows[ob_idx]

                    if (fvg_ok
                            and not overlaps_bull(top_lvl, bot_lvl, ob_idx)
                            and is_bull_fresh(ob_idx, top_lvl, bot_lvl, check_bar)):
                        placed_bull_obs.append((top_lvl, bot_lvl, ob_idx))
                        confirmed_obs.append({
                            'type':         'BULL',
                            'stage':        'CONFIRMED',
                            'ob_idx':       ob_idx,
                            'fractal_idx':  f_idx,
                            'fractal_level':f_level,
                            'break_idx':    check_bar,
                            'break_close':  closes[check_bar],
                            'ob_open':      opens[ob_idx],
                            'ob_high':      highs[ob_idx],
                            'ob_low':       lows[ob_idx],
                            'ob_close':     closes[ob_idx],
                            'ob_time':      times[ob_idx],
                            'break_time':   times[check_bar],
                            'top_level':    top_lvl,
                            'bot_level':    bot_lvl,
                            'break_age':    last_closed_idx - check_bar,
                        })
                active_fractal_high = None

        # BEARISH BREAK
        if active_fractal_low is not None:
            f_idx, f_level = active_fractal_low
            if check_bar > f_idx + side and closes[check_bar] < f_level:
                ob_idx = -1
                search_start = max(0, check_bar - OB_SEARCH_LOOKBACK)
                for k in range(check_bar - 1, search_start - 1, -1):
                    if is_bullish(opens[k], closes[k]):
                        ob_idx = k
                        break

                if ob_idx >= 0:
                    fvg_ok = True
                    if REQUIRE_FVG:
                        fvg_ok = has_bearish_fvg_after(
                            highs, lows, ob_idx, MAX_OB_TO_FVG_BARS, n
                        ) >= 0

                    top_lvl = highs[ob_idx]
                    bot_lvl = opens[ob_idx]

                    if (fvg_ok
                            and not overlaps_bear(top_lvl, bot_lvl, ob_idx)
                            and is_bear_fresh(ob_idx, top_lvl, bot_lvl, check_bar)):
                        placed_bear_obs.append((top_lvl, bot_lvl, ob_idx))
                        confirmed_obs.append({
                            'type':         'BEAR',
                            'stage':        'CONFIRMED',
                            'ob_idx':       ob_idx,
                            'fractal_idx':  f_idx,
                            'fractal_level':f_level,
                            'break_idx':    check_bar,
                            'break_close':  closes[check_bar],
                            'ob_open':      opens[ob_idx],
                            'ob_high':      highs[ob_idx],
                            'ob_low':       lows[ob_idx],
                            'ob_close':     closes[ob_idx],
                            'ob_time':      times[ob_idx],
                            'break_time':   times[check_bar],
                            'top_level':    top_lvl,
                            'bot_level':    bot_lvl,
                            'break_age':    last_closed_idx - check_bar,
                        })
                active_fractal_low = None

    # PRE-ALERTS on live candle
    live_close = closes[live_idx]
    live_time  = times[live_idx]

    if active_fractal_high is not None:
        f_idx, f_level = active_fractal_high
        if live_idx > f_idx + side and live_close > f_level:
            ob_idx = -1
            search_start = max(0, live_idx - OB_SEARCH_LOOKBACK)
            for k in range(live_idx - 1, search_start - 1, -1):
                if is_bearish(opens[k], closes[k]):
                    ob_idx = k
                    break
            if ob_idx >= 0:
                fvg_ok = True
                if REQUIRE_FVG:
                    fvg_ok = has_bullish_fvg_after(
                        highs, lows, ob_idx, MAX_OB_TO_FVG_BARS, n
                    ) >= 0
                top_lvl = opens[ob_idx]
                bot_lvl = lows[ob_idx]
                if (fvg_ok
                        and not overlaps_bull(top_lvl, bot_lvl, ob_idx)
                        and is_bull_fresh(ob_idx, top_lvl, bot_lvl, live_idx)):
                    prealert_obs.append({
                        'type':         'BULL',
                        'stage':        'PREALERT',
                        'ob_idx':       ob_idx,
                        'fractal_idx':  f_idx,
                        'fractal_level':f_level,
                        'live_close':   live_close,
                        'live_time':    live_time,
                        'ob_open':      opens[ob_idx],
                        'ob_high':      highs[ob_idx],
                        'ob_low':       lows[ob_idx],
                        'ob_time':      times[ob_idx],
                        'top_level':    top_lvl,
                        'bot_level':    bot_lvl,
                    })

    if active_fractal_low is not None:
        f_idx, f_level = active_fractal_low
        if live_idx > f_idx + side and live_close < f_level:
            ob_idx = -1
            search_start = max(0, live_idx - OB_SEARCH_LOOKBACK)
            for k in range(live_idx - 1, search_start - 1, -1):
                if is_bullish(opens[k], closes[k]):
                    ob_idx = k
                    break
            if ob_idx >= 0:
                fvg_ok = True
                if REQUIRE_FVG:
                    fvg_ok = has_bearish_fvg_after(
                        highs, lows, ob_idx, MAX_OB_TO_FVG_BARS, n
                    ) >= 0
                top_lvl = highs[ob_idx]
                bot_lvl = opens[ob_idx]
                if (fvg_ok
                        and not overlaps_bear(top_lvl, bot_lvl, ob_idx)
                        and is_bear_fresh(ob_idx, top_lvl, bot_lvl, live_idx)):
                    prealert_obs.append({
                        'type':         'BEAR',
                        'stage':        'PREALERT',
                        'ob_idx':       ob_idx,
                        'fractal_idx':  f_idx,
                        'fractal_level':f_level,
                        'live_close':   live_close,
                        'live_time':    live_time,
                        'ob_open':      opens[ob_idx],
                        'ob_high':      highs[ob_idx],
                        'ob_low':       lows[ob_idx],
                        'ob_time':      times[ob_idx],
                        'top_level':    top_lvl,
                        'bot_level':    bot_lvl,
                    })

    fresh_confirmed = [
        ob for ob in confirmed_obs
        if ob['break_age'] <= MAX_BREAK_AGE_BARS
    ]

    return fresh_confirmed, prealert_obs


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


def fmt_compact(coin: str, tf: str, ob: dict) -> str:
    is_bull = ob['type'] == 'BULL'
    stage   = ob['stage']
    if stage == 'CONFIRMED':
        emoji = '🟢' if is_bull else '🔴'
        label = 'Bull OB' if is_bull else 'Bear OB'
        prefix = ''
    else:
        emoji = '🟡'
        label = 'POTENTIAL Bull OB' if is_bull else 'POTENTIAL Bear OB'
        prefix = '⚠️ PRE-ALERT — '
    top = ob['top_level']
    bot = ob['bot_level']
    return (
        f"{emoji} {prefix}{coin} {tf} | {label}\n"
        f"Zone: {bot:,.4f} – {top:,.4f}"
    )


def send_startup_msg():
    send_msg(
        f"🤖 <b>OB Alert Bot — LIVE (STRICT)</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Coins:</b> {', '.join(COINS)}\n"
        f"<b>Timeframes:</b> {', '.join(TIMEFRAMES)}\n\n"
        f"<b>Mode:</b> Nephew_Sam_ replica\n"
        f"<b>Fractal:</b> {FRACTAL_TYPE}-bar\n"
        f"<b>FVG required:</b> {REQUIRE_FVG}\n"
        f"<b>OB lookback:</b> {OB_SEARCH_LOOKBACK}\n"
        f"<b>Skip overlap:</b> {SKIP_OVERLAP}\n"
        f"<b>Fresh only:</b> {FRESH_ONLY}\n"
        f"<b>Max break age:</b> {MAX_BREAK_AGE_BARS} candles\n\n"
        f"<b>Alerts:</b>\n"
        f"  🟡 PRE-ALERT — fires only LATE in candle:\n"
        f"     • 15m: minute 8–15 of candle\n"
        f"     • 1H:  minute 40–60 of candle\n"
        f"  🟢 CONFIRMED — within last {MAX_BREAK_AGE_BARS} closed candles\n\n"
        f"<b>Scan:</b> Every 1 minute\n"
        f"<b>Cooldown:</b> 30 min per type\n\n"
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

                confirmed_obs, prealert_obs = detect_obs(df)

                # CONFIRMED alerts
                confirmed_obs.sort(key=lambda o: o['break_idx'], reverse=True)
                count_conf = 0
                for ob in confirmed_obs[:MAX_OBS_PER_COIN]:
                    sig = (
                        f"CONF_{coin_name}_{tf_label}_{ob['type']}_"
                        f"{pd.Timestamp(ob['ob_time']).isoformat()}_"
                        f"{pd.Timestamp(ob['break_time']).isoformat()}"
                    )
                    if sig in sent_signatures:
                        continue
                    ck = f"{coin_name}_{tf_label}_{ob['type']}"
                    if (time.time() - last_alerts.get(ck, 0)
                            < COOLDOWN_SECONDS):
                        continue
                    send_msg(fmt_compact(coin_name, tf_label, ob))
                    sent_signatures.add(sig)
                    last_alerts[ck] = time.time()
                    count_conf += 1
                    logger.info(
                        f"✅ CONFIRMED {ob['type']} OB {coin_name} {tf_label} | "
                        f"break_age={ob['break_age']} | "
                        f"zone[{ob['bot_level']:.4f}–{ob['top_level']:.4f}]"
                    )

                # PRE-ALERTS (only after candle reaches min age)
                count_pre = 0
                if len(df) > 0:
                    is_old_enough, candle_age = is_candle_old_enough(df, tf_label)

                    if not is_old_enough:
                        if prealert_obs:
                            min_age = PREALERT_MIN_AGE_MINUTES.get(tf_label, 0)
                            logger.info(
                                f"   ⏳ {coin_name} {tf_label}: "
                                f"{len(prealert_obs)} pre-alert(s) WAITING "
                                f"(candle age {candle_age}m < {min_age}m)"
                            )
                    else:
                        live_candle_time = df['time'].iloc[-1]
                        for ob in prealert_obs[:MAX_OBS_PER_COIN]:
                            sig = (
                                f"PRE_{coin_name}_{tf_label}_{ob['type']}_"
                                f"{pd.Timestamp(live_candle_time).isoformat()}"
                            )
                            if sig in prealert_signatures:
                                continue
                            send_msg(fmt_compact(coin_name, tf_label, ob))
                            prealert_signatures.add(sig)
                            count_pre += 1
                            logger.info(
                                f"⚠️ PRE-ALERT {ob['type']} OB {coin_name} {tf_label} | "
                                f"candle_age={candle_age}m | "
                                f"live_close={ob['live_close']:.4f} | "
                                f"zone[{ob['bot_level']:.4f}–{ob['top_level']:.4f}]"
                            )

                if count_conf or count_pre:
                    logger.info(
                        f"   → {coin_name} {tf_label}: "
                        f"{count_conf} confirmed, {count_pre} pre-alerts"
                    )

                time.sleep(0.2)

        if len(sent_signatures) > 5000:
            for s in list(sent_signatures)[:2500]:
                sent_signatures.discard(s)
        if len(prealert_signatures) > 5000:
            for s in list(prealert_signatures)[:2500]:
                prealert_signatures.discard(s)

        logger.info(
            f"✅ Scan complete — "
            f"{len(sent_signatures)} confirmed | "
            f"{len(prealert_signatures)} pre-alerts (lifetime)"
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
    return (
        f"<h2>🤖 Order Block Alert Bot (STRICT)</h2>"
        f"<p><b>Status:</b> ✅ Running</p>"
        f"<p><b>Time UTC:</b> {now.strftime('%Y-%m-%d %H:%M:%S')}</p>"
        f"<p><b>Mode:</b> Pine Script replica + close confirm</p>"
        f"<p><b>Data:</b> Binance.US | <b>Scan:</b> 1 min</p>"
        f"<p><b>Max break age:</b> {MAX_BREAK_AGE_BARS} candles</p>"
        f"<p><b>Pre-alert min age:</b> 15m={PREALERT_MIN_AGE_MINUTES['15m']}m, "
        f"1h={PREALERT_MIN_AGE_MINUTES['1h']}m</p>"
        f"<p><b>Confirmed sent:</b> {len(sent_signatures)}</p>"
        f"<p><b>Pre-alerts sent:</b> {len(prealert_signatures)}</p>"
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
    return {
        'status':           'running',
        'mode':             'STRICT + late pre-alerts',
        'time_utc':         datetime.now(timezone.utc).isoformat(),
        'coins':            list(COINS.keys()),
        'timeframes':       list(TIMEFRAMES.keys()),
        'confirmed_sent':   len(sent_signatures),
        'prealerts_sent':   len(prealert_signatures),
        'cooldowns':        len(last_alerts),
        'settings': {
            'fractal_type':            FRACTAL_TYPE,
            'require_fvg':             REQUIRE_FVG,
            'max_ob_to_fvg_bars':      MAX_OB_TO_FVG_BARS,
            'ob_search_lookback':      OB_SEARCH_LOOKBACK,
            'skip_overlap':            SKIP_OVERLAP,
            'overlap_window':          OVERLAP_WINDOW,
            'fresh_only':              FRESH_ONLY,
            'max_obs_per_coin':        MAX_OBS_PER_COIN,
            'max_break_age_bars':      MAX_BREAK_AGE_BARS,
            'prealert_min_age_minutes':PREALERT_MIN_AGE_MINUTES,
        },
    }


@app.route('/scan_now')
def scan_now_route():
    threading.Thread(target=scan_all, daemon=True).start()
    return 'Scan triggered!', 200


@app.route('/reset')
def reset():
    sent_signatures.clear()
    prealert_signatures.clear()
    last_alerts.clear()
    return 'Reset done!', 200


# ============================================================
# STARTUP
# ============================================================

def start_bot():
    logger.info("Order Block Bot starting (STRICT + late pre-alerts)…")
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
