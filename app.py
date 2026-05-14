# ============================================================
# ORDER BLOCK ALERT BOT
# Coins: BTC, ETH, SOL  |  Timeframes: 15m, 1H
# Logic: Nephew_Sam_ replica + close confirmation
# Two-stage alerts:
#   🟡 PRE-ALERT  → live candle currently breaking fractal
#   🟢 CONFIRMED  → candle closed beyond fractal
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

# ── OB Detection Settings (matches Pine Script defaults) ─────
FRACTAL_TYPE        = 3      # 3-bar fractal (1 left, 1 right)
REQUIRE_FVG         = True
MAX_OB_TO_FVG_BARS  = 3
OB_SEARCH_LOOKBACK  = 5
SKIP_OVERLAP        = True
OVERLAP_WINDOW      = 20
FRESH_ONLY          = True

# ── Cooldown ─────────────────────────────────────────────────
COOLDOWN_SECONDS    = 30 * 60      # 30 min same coin/tf/type
MAX_OBS_PER_COIN    = 3            # Max OBs per scan per coin

# ── Alert state ──────────────────────────────────────────────
last_alerts          = {}
sent_signatures      = set()       # confirmed dedup (permanent)
prealert_signatures  = set()       # pre-alert dedup (per candle)

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
    """
    Fetch OHLC including the CURRENTLY FORMING candle (last row).
    The last row is live and updates with each tick.
    """
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
        df['time']       = pd.to_datetime(df['time'],       unit='ms', utc=True)
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].astype(float)
        return df[['time', 'open', 'high', 'low', 'close', 'close_time']].reset_index(drop=True)
    except Exception as e:
        logger.error(f"fetch_klines {symbol}/{interval}: {e}")
        return None


# ============================================================
# FRACTAL DETECTION (matches Pine Script)
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

def has_bullish_fvg_after(highs, lows, ob_idx: int,
                           max_bars: int, n: int) -> int:
    """Bullish FVG: low[i+2] > high[i]. Returns index of FVG or -1."""
    end = min(n - 2, ob_idx + max_bars)
    for i in range(ob_idx, end + 1):
        if i + 2 >= n: break
        if lows[i + 2] > highs[i]:
            return i + 1
    return -1


def has_bearish_fvg_after(highs, lows, ob_idx: int,
                           max_bars: int, n: int) -> int:
    """Bearish FVG: high[i+2] < low[i]. Returns index of FVG or -1."""
    end = min(n - 2, ob_idx + max_bars)
    for i in range(ob_idx, end + 1):
        if i + 2 >= n: break
        if highs[i + 2] < lows[i]:
            return i + 1
    return -1


# ============================================================
# OB DETECTION — Nephew_Sam_ replica
# ============================================================

def is_bullish(o: float, c: float) -> bool:
    return c > o


def is_bearish(o: float, c: float) -> bool:
    return c < o


def detect_obs(df: pd.DataFrame, include_live: bool = True) -> tuple[list, list]:
    """
    Detect Order Blocks following the Pine Script logic exactly.

    Returns (confirmed_obs, prealert_obs)

    confirmed_obs: OBs where a CLOSED candle has broken a fractal level
    prealert_obs:  OBs where the CURRENTLY FORMING candle is breaking a level
                   (price hasn't closed yet, but is currently beyond)
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

    # The last row is the LIVE forming candle
    live_idx = n - 1   # currently forming candle index

    # Find all fractal pivots (must exclude live candle and those too close to edge)
    fractal_highs = []   # [(idx, level)]
    fractal_lows  = []
    for i in range(side, n - side):
        # Don't use fractals where the right side includes the live candle
        # because that candle isn't closed yet
        if i + side >= live_idx:
            continue
        if is_fractal_high(highs, i, side):
            fractal_highs.append((i, highs[i]))
        if is_fractal_low(lows, i, side):
            fractal_lows.append((i, lows[i]))

    confirmed_obs = []
    prealert_obs  = []

    # Track existing OB zones to check overlap
    placed_bull_obs = []   # list of (top, bot, bar_idx)
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
        """Check no candle between ob_bar+1 and until_bar-1 closed below bot."""
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

    # ─────────────────────────────────────────────────────
    # BULLISH OB: fractal HIGH broken by CLOSE above
    # ─────────────────────────────────────────────────────
    for f_idx, f_level in fractal_highs:
        # Look for first candle AFTER f_idx that closes above f_level
        # (using only CLOSED candles — exclude live)
        break_idx = -1
        for j in range(f_idx + side + 1, live_idx):   # exclude live
            if closes[j] > f_level:
                break_idx = j
                break

        # ── CONFIRMED OB on closed candle ────────────────
        if break_idx > 0:
            # Look back from break_idx for last RED candle = OB
            ob_idx = -1
            search_start = max(0, break_idx - OB_SEARCH_LOOKBACK)
            for k in range(break_idx - 1, search_start - 1, -1):
                if is_bearish(opens[k], closes[k]):
                    ob_idx = k
                    break
            if ob_idx < 0: continue

            # Check FVG
            if REQUIRE_FVG:
                fvg_idx = has_bullish_fvg_after(
                    highs, lows, ob_idx, MAX_OB_TO_FVG_BARS, n
                )
                if fvg_idx < 0: continue

            top_lvl = opens[ob_idx]      # candle OPEN
            bot_lvl = lows[ob_idx]       # wick LOW

            # Quality filters
            if overlaps_bull(top_lvl, bot_lvl, ob_idx): continue
            if not is_bull_fresh(ob_idx, top_lvl, bot_lvl, break_idx): continue

            placed_bull_obs.append((top_lvl, bot_lvl, ob_idx))

            confirmed_obs.append({
                'type':         'BULL',
                'stage':        'CONFIRMED',
                'ob_idx':       ob_idx,
                'fractal_idx':  f_idx,
                'fractal_level':f_level,
                'break_idx':    break_idx,
                'break_close':  closes[break_idx],
                'ob_open':      opens[ob_idx],
                'ob_high':      highs[ob_idx],
                'ob_low':       lows[ob_idx],
                'ob_close':     closes[ob_idx],
                'ob_time':      times[ob_idx],
                'break_time':   times[break_idx],
                'top_level':    top_lvl,
                'bot_level':    bot_lvl,
            })
            continue   # move to next fractal

        # ── PRE-ALERT: live candle currently above f_level ─
        if include_live and closes[live_idx] > f_level:
            # Look back from live for last RED candle = potential OB
            ob_idx = -1
            search_start = max(0, live_idx - OB_SEARCH_LOOKBACK)
            for k in range(live_idx - 1, search_start - 1, -1):
                if is_bearish(opens[k], closes[k]):
                    ob_idx = k
                    break
            if ob_idx < 0: continue

            if REQUIRE_FVG:
                fvg_idx = has_bullish_fvg_after(
                    highs, lows, ob_idx, MAX_OB_TO_FVG_BARS, n
                )
                if fvg_idx < 0: continue

            top_lvl = opens[ob_idx]
            bot_lvl = lows[ob_idx]

            if overlaps_bull(top_lvl, bot_lvl, ob_idx): continue
            if not is_bull_fresh(ob_idx, top_lvl, bot_lvl, live_idx): continue

            prealert_obs.append({
                'type':         'BULL',
                'stage':        'PREALERT',
                'ob_idx':       ob_idx,
                'fractal_idx':  f_idx,
                'fractal_level':f_level,
                'live_close':   closes[live_idx],
                'live_time':    times[live_idx],
                'ob_open':      opens[ob_idx],
                'ob_high':      highs[ob_idx],
                'ob_low':       lows[ob_idx],
                'ob_time':      times[ob_idx],
                'top_level':    top_lvl,
                'bot_level':    bot_lvl,
            })

    # ─────────────────────────────────────────────────────
    # BEARISH OB: fractal LOW broken by CLOSE below
    # ─────────────────────────────────────────────────────
    for f_idx, f_level in fractal_lows:
        break_idx = -1
        for j in range(f_idx + side + 1, live_idx):
            if closes[j] < f_level:
                break_idx = j
                break

        if break_idx > 0:
            ob_idx = -1
            search_start = max(0, break_idx - OB_SEARCH_LOOKBACK)
            for k in range(break_idx - 1, search_start - 1, -1):
                if is_bullish(opens[k], closes[k]):
                    ob_idx = k
                    break
            if ob_idx < 0: continue

            if REQUIRE_FVG:
                fvg_idx = has_bearish_fvg_after(
                    highs, lows, ob_idx, MAX_OB_TO_FVG_BARS, n
                )
                if fvg_idx < 0: continue

            top_lvl = highs[ob_idx]      # wick HIGH
            bot_lvl = opens[ob_idx]      # candle OPEN

            if overlaps_bear(top_lvl, bot_lvl, ob_idx): continue
            if not is_bear_fresh(ob_idx, top_lvl, bot_lvl, break_idx): continue

            placed_bear_obs.append((top_lvl, bot_lvl, ob_idx))

            confirmed_obs.append({
                'type':         'BEAR',
                'stage':        'CONFIRMED',
                'ob_idx':       ob_idx,
                'fractal_idx':  f_idx,
                'fractal_level':f_level,
                'break_idx':    break_idx,
                'break_close':  closes[break_idx],
                'ob_open':      opens[ob_idx],
                'ob_high':      highs[ob_idx],
                'ob_low':       lows[ob_idx],
                'ob_close':     closes[ob_idx],
                'ob_time':      times[ob_idx],
                'break_time':   times[break_idx],
                'top_level':    top_lvl,
                'bot_level':    bot_lvl,
            })
            continue

        if include_live and closes[live_idx] < f_level:
            ob_idx = -1
            search_start = max(0, live_idx - OB_SEARCH_LOOKBACK)
            for k in range(live_idx - 1, search_start - 1, -1):
                if is_bullish(opens[k], closes[k]):
                    ob_idx = k
                    break
            if ob_idx < 0: continue

            if REQUIRE_FVG:
                fvg_idx = has_bearish_fvg_after(
                    highs, lows, ob_idx, MAX_OB_TO_FVG_BARS, n
                )
                if fvg_idx < 0: continue

            top_lvl = highs[ob_idx]
            bot_lvl = opens[ob_idx]

            if overlaps_bear(top_lvl, bot_lvl, ob_idx): continue
            if not is_bear_fresh(ob_idx, top_lvl, bot_lvl, live_idx): continue

            prealert_obs.append({
                'type':         'BEAR',
                'stage':        'PREALERT',
                'ob_idx':       ob_idx,
                'fractal_idx':  f_idx,
                'fractal_level':f_level,
                'live_close':   closes[live_idx],
                'live_time':    times[live_idx],
                'ob_open':      opens[ob_idx],
                'ob_high':      highs[ob_idx],
                'ob_low':       lows[ob_idx],
                'ob_time':      times[ob_idx],
                'top_level':    top_lvl,
                'bot_level':    bot_lvl,
            })

    return confirmed_obs, prealert_obs


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
    """Compact alert format as you specified."""
    is_bull = ob['type'] == 'BULL'
    stage   = ob['stage']

    if stage == 'CONFIRMED':
        emoji = '🟢' if is_bull else '🔴'
        label = 'Bull OB' if is_bull else 'Bear OB'
        prefix = ''
    else:   # PREALERT
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
        f"🤖 <b>OB Alert Bot — LIVE</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Coins:</b> {', '.join(COINS)}\n"
        f"<b>Timeframes:</b> {', '.join(TIMEFRAMES)}\n\n"
        f"<b>Mode:</b> Nephew_Sam_ replica\n"
        f"<b>Fractal:</b> {FRACTAL_TYPE}-bar\n"
        f"<b>FVG required:</b> {REQUIRE_FVG}\n"
        f"<b>OB lookback:</b> {OB_SEARCH_LOOKBACK}\n"
        f"<b>Skip overlap:</b> {SKIP_OVERLAP}\n"
        f"<b>Fresh only:</b> {FRESH_ONLY}\n\n"
        f"<b>Alerts:</b>\n"
        f"  🟡 PRE-ALERT — live candle breaking\n"
        f"  🟢 CONFIRMED — close confirmed\n\n"
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

                confirmed_obs, prealert_obs = detect_obs(df, include_live=True)

                # ── Send CONFIRMED alerts ──────────────────
                count_confirmed = 0
                # Sort newest first, limit to MAX_OBS_PER_COIN
                confirmed_obs.sort(key=lambda o: o['break_idx'], reverse=True)
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
                    count_confirmed += 1
                    logger.info(
                        f"✅ CONFIRMED {ob['type']} OB {coin_name} {tf_label} | "
                        f"zone[{ob['bot_level']:.4f}–{ob['top_level']:.4f}]"
                    )

                # ── Send PRE-ALERTS (max 1 per live candle) ─
                count_prealert = 0
                # Pre-alerts are deduped per LIVE candle
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
                    count_prealert += 1
                    logger.info(
                        f"⚠️ PRE-ALERT {ob['type']} OB {coin_name} {tf_label} | "
                        f"live close={ob['live_close']:.4f} | "
                        f"zone[{ob['bot_level']:.4f}–{ob['top_level']:.4f}]"
                    )

                if count_confirmed or count_prealert:
                    logger.info(
                        f"   → {coin_name} {tf_label}: "
                        f"{count_confirmed} confirmed, {count_prealert} pre-alerts"
                    )

                time.sleep(0.2)

        # Trim memory
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
        f"<h2>🤖 Order Block Alert Bot</h2>"
        f"<p><b>Status:</b> ✅ Running</p>"
        f"<p><b>Time UTC:</b> {now.strftime('%Y-%m-%d %H:%M:%S')}</p>"
        f"<p><b>Mode:</b> Nephew_Sam_ replica + close confirm</p>"
        f"<p><b>Data source:</b> Binance.US</p>"
        f"<p><b>Scan interval:</b> 1 minute</p>"
        f"<p><b>Settings:</b> {FRACTAL_TYPE}-bar fractal | "
        f"FVG: {REQUIRE_FVG} | "
        f"Lookback: {OB_SEARCH_LOOKBACK} | "
        f"Overlap skip: {SKIP_OVERLAP} | "
        f"Fresh only: {FRESH_ONLY}</p>"
        f"<p><b>Confirmed alerts sent:</b> {len(sent_signatures)}</p>"
        f"<p><b>Pre-alerts sent:</b> {len(prealert_signatures)}</p>"
        f"<p><b>Active cooldowns:</b> {len(last_alerts)}</p>"
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
        'status':            'running',
        'mode':              'Nephew_Sam_ replica + close confirm',
        'time_utc':          datetime.now(timezone.utc).isoformat(),
        'coins':             list(COINS.keys()),
        'timeframes':        list(TIMEFRAMES.keys()),
        'confirmed_sent':    len(sent_signatures),
        'prealerts_sent':    len(prealert_signatures),
        'cooldowns':         len(last_alerts),
        'settings': {
            'fractal_type':         FRACTAL_TYPE,
            'require_fvg':          REQUIRE_FVG,
            'max_ob_to_fvg_bars':   MAX_OB_TO_FVG_BARS,
            'ob_search_lookback':   OB_SEARCH_LOOKBACK,
            'skip_overlap':         SKIP_OVERLAP,
            'overlap_window':       OVERLAP_WINDOW,
            'fresh_only':           FRESH_ONLY,
            'max_obs_per_coin':     MAX_OBS_PER_COIN,
        },
    }


@app.route('/scan_now')
def scan_now_route():
    threading.Thread(target=scan_all, daemon=True).start()
    return 'OB scan triggered! Check logs.', 200


@app.route('/reset')
def reset():
    sent_signatures.clear()
    prealert_signatures.clear()
    last_alerts.clear()
    return 'Reset done — all state cleared.', 200


# ============================================================
# STARTUP
# ============================================================

def start_bot():
    logger.info("Order Block Bot starting…")

    try:
        send_startup_msg()
    except Exception as e:
        logger.error(f"Startup msg error: {e}")

    scheduler = BackgroundScheduler(timezone='UTC')
    scheduler.add_job(
        scan_all,
        trigger='cron',
        minute='*',                 # Every 1 minute
        id='ob_scan_job',
        max_instances=1,
        coalesce=True,
        misfire_grace_time=20,
    )
    scheduler.start()
    logger.info("Scheduler started — OB scan every 1 minute")

    # Initial scan
    threading.Thread(target=scan_all, daemon=True).start()


threading.Thread(target=start_bot, daemon=True).start()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
