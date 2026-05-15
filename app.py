# ============================================================
# ORDER BLOCK ALERT BOT — STRUCTURAL CHANGE + MASSIVE BREAK
# Coins: BTC, ETH, SOL  |  Timeframes: 15m, 1H
#
# DETECTION:
#   - Streak of 2+ same-color candles
#   - Followed by opposite-color candle with body ≥ 1.5x avg
#   - OB = the last candle of the streak (right before structural change)
#   - Yellow lines drawn IMMEDIATELY (no FVG lag)
#
# PRE-ALERT:
#   - Live candle's price breaks pending OB body
#   - Once per live candle (no spam on retraces)
#
# CONFIRMED:
#   - Candle CLOSES beyond OB body
#   - AND that candle's body ≥ 2.5x avg (MASSIVE break)
#
# FAILED:
#   - Any candle closes opposite past OB wick (no size required)
#
# Data: Binance.US | Scan: every 1 min
# ============================================================

import os
import time
import logging
import threading
import requests
import pandas as pd
import numpy as np
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
}
TIMEFRAMES = {
    '15m': '15m',
    '1h':  '1h',
}

BINANCE_US_BASE  = 'https://api.binance.us/api/v3'
BINANCE_INTERVAL = {'15m': '15m', '1h': '1h'}

# ── Detection Settings (match Pine Script) ───────────────────
STREAK_LENGTH        = 2      # min same-color streak before structural change
STRUCT_BODY_MULT     = 1.5    # structural change candle body multiplier
BREAK_BODY_MULT      = 2.5    # confirmation break candle body multiplier (MASSIVE)
AVG_BODY_PERIOD      = 14     # avg body lookback

# ── Quality Filters ──────────────────────────────────────────
SKIP_OVERLAP         = True
OVERLAP_WINDOW       = 20

# ── Wall-clock freshness ─────────────────────────────────────
MAX_CLOSED_CANDLE_AGE_MINUTES = {'15m': 16, '1h': 62}

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)
app    = Flask(__name__)

try:
    bot = telebot.TeleBot(TELEGRAM_TOKEN, threaded=False)
    logger.info("Telegram bot initialized")
except Exception as e:
    logger.error(f"Telegram init error: {e}")
    bot = None


# ============================================================
# STATE
# ============================================================

# pending_obs[key] = {coin, tf, type, ob_body_top, ob_body_bot,
#                     ob_wick_high, ob_wick_low, ob_time,
#                     struct_change_time, created_close_time}
pending_obs    = {}
pending_lock   = threading.Lock()
prealert_sent  = set()      # dedup pre-alerts per live candle
finalized_sent = set()      # dedup confirm/fail per OB key


# ============================================================
# BINANCE
# ============================================================

def fetch_klines(symbol, interval, limit=500):
    try:
        resp = requests.get(
            f"{BINANCE_US_BASE}/klines",
            params={'symbol': symbol, 'interval': interval, 'limit': limit},
            timeout=15)
        if resp.status_code != 200:
            logger.error(f"Binance.US {resp.status_code} — {symbol}/{interval}")
            return None
        klines = resp.json()
        if not klines or isinstance(klines, dict): return None
        df = pd.DataFrame(klines, columns=[
            'time','open','high','low','close','volume',
            'close_time','qav','num_trades',
            'taker_base','taker_quote','ignore'])
        df['time']       = pd.to_datetime(df['time'],       unit='ms', utc=True)
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)
        for c in ['open','high','low','close']:
            df[c] = df[c].astype(float)
        return df[['time','open','high','low','close','close_time']].reset_index(drop=True)
    except Exception as e:
        logger.error(f"fetch_klines {symbol}/{interval}: {e}")
        return None


# ============================================================
# HELPERS
# ============================================================

def closed_candle_too_old(df, tf):
    if df is None or len(df) < 2: return True
    close_time = df['close_time'].iloc[-2].to_pydatetime()
    age_min = (datetime.now(timezone.utc) - close_time).total_seconds() / 60
    threshold = MAX_CLOSED_CANDLE_AGE_MINUTES.get(tf, 999999)
    return age_min > threshold


def is_bullish(o, c): return c > o
def is_bearish(o, c): return c < o


def get_avg_body(opens, closes, end_idx, period):
    """Average body of `period` candles ending BEFORE end_idx."""
    start = max(0, end_idx - period)
    bodies = np.abs(closes[start:end_idx] - opens[start:end_idx])
    if len(bodies) == 0: return 0.0
    return float(np.mean(bodies))


def is_bull_streak(opens, closes, end_idx, n):
    """True if candles [end_idx-n .. end_idx-1] are all bullish."""
    for k in range(1, n + 1):
        idx = end_idx - k
        if idx < 0: return False
        if not is_bullish(opens[idx], closes[idx]): return False
    return True


def is_bear_streak(opens, closes, end_idx, n):
    for k in range(1, n + 1):
        idx = end_idx - k
        if idx < 0: return False
        if not is_bearish(opens[idx], closes[idx]): return False
    return True


# ============================================================
# DETECT NEW OB FROM JUST-CLOSED CANDLE
# ============================================================

def detect_new_ob(df, candle_idx, coin, tf):
    """
    Check if the candle at candle_idx is a structural change candle.
    If yes, return the OB (which is the last candle of the streak).
    Also return whether this OB is INSTANT CONFIRMED (close beyond + massive).
    """
    n = len(df)
    if candle_idx < AVG_BODY_PERIOD + STREAK_LENGTH + 1:
        return None

    opens, highs, lows, closes, times = (
        df['open'].values, df['high'].values,
        df['low'].values,  df['close'].values, df['time'].values)

    avg_body = get_avg_body(opens, closes, candle_idx, AVG_BODY_PERIOD)
    if avg_body == 0: return None

    candle_open  = opens[candle_idx]
    candle_close = closes[candle_idx]
    candle_body  = abs(candle_close - candle_open)

    # Structural change body must be ≥ 1.5x avg
    struct_ok = candle_body >= avg_body * STRUCT_BODY_MULT
    if not struct_ok: return None

    # The candle directly before the structural change = the OB
    ob_idx = candle_idx - 1
    if ob_idx < 0: return None

    is_bull_change = is_bullish(candle_open, candle_close)
    is_bear_change = is_bearish(candle_open, candle_close)

    # ── BULL OB: bear streak of N candles ENDING at ob_idx ──
    # The OB candle is the last bear; we need bear streak of N candles total
    # Streak: candles [ob_idx - N + 1 .. ob_idx] all bearish
    if is_bull_change:
        bear_streak = True
        for k in range(0, STREAK_LENGTH):
            idx = ob_idx - k
            if idx < 0:
                bear_streak = False
                break
            if not is_bearish(opens[idx], closes[idx]):
                bear_streak = False
                break
        if not bear_streak: return None

        # OB = the bear candle right before structural change
        body_top  = opens[ob_idx]    # bear candle: open is higher
        body_bot  = closes[ob_idx]   # close is lower
        wick_high = highs[ob_idx]
        wick_low  = lows[ob_idx]

        # Check if structural change candle was MASSIVE enough for instant confirm
        is_massive = candle_body >= avg_body * BREAK_BODY_MULT
        instant_confirmed = is_massive and candle_close > body_top

        return {
            'key': f"{coin}_{tf}_BULL_{pd.Timestamp(times[ob_idx]).isoformat()}",
            'coin': coin, 'tf': tf, 'type': 'BULL',
            'ob_body_top': body_top, 'ob_body_bot': body_bot,
            'ob_wick_high': wick_high, 'ob_wick_low': wick_low,
            'ob_time': times[ob_idx],
            'struct_change_time': times[candle_idx],
            'struct_change_close': candle_close,
            'instant_confirmed': instant_confirmed,
            'created_idx': candle_idx,
        }

    # ── BEAR OB: bull streak ENDING at ob_idx ──
    if is_bear_change:
        bull_streak = True
        for k in range(0, STREAK_LENGTH):
            idx = ob_idx - k
            if idx < 0:
                bull_streak = False
                break
            if not is_bullish(opens[idx], closes[idx]):
                bull_streak = False
                break
        if not bull_streak: return None

        body_top  = closes[ob_idx]   # bull candle: close is higher
        body_bot  = opens[ob_idx]    # open is lower
        wick_high = highs[ob_idx]
        wick_low  = lows[ob_idx]

        is_massive = candle_body >= avg_body * BREAK_BODY_MULT
        instant_confirmed = is_massive and candle_close < body_bot

        return {
            'key': f"{coin}_{tf}_BEAR_{pd.Timestamp(times[ob_idx]).isoformat()}",
            'coin': coin, 'tf': tf, 'type': 'BEAR',
            'ob_body_top': body_top, 'ob_body_bot': body_bot,
            'ob_wick_high': wick_high, 'ob_wick_low': wick_low,
            'ob_time': times[ob_idx],
            'struct_change_time': times[candle_idx],
            'struct_change_close': candle_close,
            'instant_confirmed': instant_confirmed,
            'created_idx': candle_idx,
        }

    return None


def overlap_check(ob, coin, tf):
    if not SKIP_OVERLAP: return False
    new_top = max(ob['ob_body_top'], ob['ob_body_bot'])
    new_bot = min(ob['ob_body_top'], ob['ob_body_bot'])
    with pending_lock:
        existing = [o for o in pending_obs.values()
                    if o['coin']==coin and o['tf']==tf and o['type']==ob['type']]
    for ex in existing:
        ex_top = max(ex['ob_body_top'], ex['ob_body_bot'])
        ex_bot = min(ex['ob_body_top'], ex['ob_body_bot'])
        if not (new_bot > ex_top or new_top < ex_bot):
            return True
    return False


# ============================================================
# CHECK PENDING OBS — CONFIRM (massive break) or FAIL (any close)
# ============================================================

def check_pending(coin, tf, df):
    if df is None or len(df) < 2: return [], []
    confirmed, failed = [], []

    n = len(df)
    last_closed_idx   = n - 2
    last_closed_open  = float(df['open'].iloc[-2])
    last_closed_close = float(df['close'].iloc[-2])
    last_closed_time  = df['time'].iloc[-2]

    # Avg body for massive check
    opens, closes = df['open'].values, df['close'].values
    avg_body = get_avg_body(opens, closes, last_closed_idx, AVG_BODY_PERIOD)
    last_body = abs(last_closed_close - last_closed_open)
    is_massive = avg_body > 0 and last_body >= avg_body * BREAK_BODY_MULT

    with pending_lock:
        keys = [k for k in pending_obs
                if pending_obs[k]['coin']==coin and pending_obs[k]['tf']==tf]
        for key in keys:
            ob = pending_obs[key]

            # Wait until next bar after creation
            if ob['struct_change_time'] >= last_closed_time:
                continue

            body_top = max(ob['ob_body_top'], ob['ob_body_bot'])
            body_bot = min(ob['ob_body_top'], ob['ob_body_bot'])

            if ob['type'] == 'BULL':
                # FAILED: any close past wick low (no body filter)
                if last_closed_close < ob['ob_wick_low']:
                    ob['fail_close'] = last_closed_close
                    ob['fail_time']  = last_closed_time
                    failed.append(ob); del pending_obs[key]
                # CONFIRMED: close above body top AND massive body
                elif last_closed_close > body_top and is_massive:
                    ob['confirm_close'] = last_closed_close
                    ob['confirm_time']  = last_closed_time
                    ob['confirm_body_mult'] = round(last_body / avg_body, 2)
                    confirmed.append(ob); del pending_obs[key]
                # else: stays pending (close beyond body but not massive, OR close inside body)
            else:  # BEAR
                if last_closed_close > ob['ob_wick_high']:
                    ob['fail_close'] = last_closed_close
                    ob['fail_time']  = last_closed_time
                    failed.append(ob); del pending_obs[key]
                elif last_closed_close < body_bot and is_massive:
                    ob['confirm_close'] = last_closed_close
                    ob['confirm_time']  = last_closed_time
                    ob['confirm_body_mult'] = round(last_body / avg_body, 2)
                    confirmed.append(ob); del pending_obs[key]

    return confirmed, failed


# ============================================================
# LIVE PRE-ALERT — checks live candle's current price vs pending OBs
# ============================================================

def check_live_prealerts(coin, tf, df):
    """Return list of OBs that are currently being broken by live price."""
    if df is None or len(df) < 1: return []
    live_close = float(df['close'].iloc[-1])
    live_time  = df['time'].iloc[-1]

    triggered = []
    with pending_lock:
        candidates = [o for o in pending_obs.values()
                      if o['coin']==coin and o['tf']==tf]
    for ob in candidates:
        body_top = max(ob['ob_body_top'], ob['ob_body_bot'])
        body_bot = min(ob['ob_body_top'], ob['ob_body_bot'])

        pre_key = (f"PRE_{coin}_{tf}_{ob['type']}_{ob['key']}_"
                   f"{pd.Timestamp(live_time).isoformat()}")
        if pre_key in prealert_sent:
            continue

        if ob['type'] == 'BULL' and live_close > body_top:
            ob['live_close'] = live_close
            ob['live_time']  = live_time
            ob['_pre_key']   = pre_key
            triggered.append(ob)
        elif ob['type'] == 'BEAR' and live_close < body_bot:
            ob['live_close'] = live_close
            ob['live_time']  = live_time
            ob['_pre_key']   = pre_key
            triggered.append(ob)

    return triggered


# ============================================================
# TELEGRAM
# ============================================================

def send_msg(text):
    try:
        if bot is None or not CHAT_ID: return
        bot.send_message(CHAT_ID, text, parse_mode='HTML')
    except Exception as e:
        logger.error(f"Telegram send error: {e}")


def fmt_pre(ob):
    is_bull = ob['type']=='BULL'
    label = 'POTENTIAL Bull OB' if is_bull else 'POTENTIAL Bear OB'
    top = max(ob['ob_body_top'], ob['ob_body_bot'])
    bot_lvl = min(ob['ob_body_top'], ob['ob_body_bot'])
    return (f"🟡 PRE-ALERT — {ob['coin']} {ob['tf']} | {label}\n"
            f"OB Zone: {bot_lvl:,.4f} – {top:,.4f}\n"
            f"Live price breaking — needs MASSIVE close to confirm")


def fmt_confirmed(ob, instant=False):
    is_bull = ob['type']=='BULL'
    emoji = '🟢' if is_bull else '🔴'
    label = 'Bull OB' if is_bull else 'Bear OB'
    top = max(ob['ob_body_top'], ob['ob_body_bot'])
    bot_lvl = min(ob['ob_body_top'], ob['ob_body_bot'])
    
    if instant:
        cls = ob['struct_change_close']
        suffix = " (instant — structural change closed massive)"
    else:
        cls = ob['confirm_close']
        mult = ob.get('confirm_body_mult', '?')
        suffix = f" (massive break: {mult}x avg body)"
    
    return (f"{emoji} CONFIRMED — {ob['coin']} {ob['tf']} | {label}{suffix}\n"
            f"OB Zone: {bot_lvl:,.4f} – {top:,.4f}\n"
            f"Closed at: {cls:,.4f}")


def fmt_failed(ob):
    is_bull = ob['type']=='BULL'
    label = 'Bull OB' if is_bull else 'Bear OB'
    top = max(ob['ob_body_top'], ob['ob_body_bot'])
    bot_lvl = min(ob['ob_body_top'], ob['ob_body_bot'])
    return (f"❌ FAILED — {ob['coin']} {ob['tf']} | {label}\n"
            f"OB was: {bot_lvl:,.4f} – {top:,.4f}\n"
            f"Closed opposite at: {ob['fail_close']:,.4f}")


def send_startup_msg():
    send_msg(
        f"🤖 <b>OB Alert Bot — STRUCTURAL + MASSIVE BREAK</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Coins:</b> {', '.join(COINS)}\n"
        f"<b>Timeframes:</b> {', '.join(TIMEFRAMES)}\n\n"
        f"<b>Detection:</b>\n"
        f"  Streak {STREAK_LENGTH}+ same color → opposite candle\n"
        f"  Structural change body ≥ {STRUCT_BODY_MULT}x avg\n"
        f"  → 🟡 Yellow lines drawn IMMEDIATELY (no FVG lag)\n\n"
        f"<b>Confirmation requires BOTH:</b>\n"
        f"  ✅ Candle CLOSES beyond OB body\n"
        f"  ✅ Body ≥ {BREAK_BODY_MULT}x avg (MASSIVE)\n\n"
        f"<b>Failure:</b> any close opposite past OB wick\n\n"
        f"<b>Filters:</b>\n"
        f"  Skip overlap: {SKIP_OVERLAP} (window {OVERLAP_WINDOW} bars)\n"
        f"  Wall-clock fresh: 15m={MAX_CLOSED_CANDLE_AGE_MINUTES['15m']}m, "
        f"1h={MAX_CLOSED_CANDLE_AGE_MINUTES['1h']}m\n\n"
        f"<i>Data: Binance.US | Scan every 1 min</i>"
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
                df = fetch_klines(coin_ticker, BINANCE_INTERVAL[tf_interval], limit=500)
                if df is None or len(df) < 50:
                    logger.warning(f"No data: {coin_name} {tf_label}")
                    continue

                last_closed_idx = len(df) - 2

                # ── 1. Check pending OBs (confirm/fail) ─────
                confirmed, failed = check_pending(coin_name, tf_label, df)
                for ob in confirmed:
                    fkey = f"FIN_{ob['key']}"
                    if fkey not in finalized_sent:
                        send_msg(fmt_confirmed(ob, instant=False))
                        finalized_sent.add(fkey)
                        logger.info(
                            f"✅ CONFIRMED {ob['type']} {coin_name} {tf_label} | "
                            f"body_mult={ob.get('confirm_body_mult','?')}x")
                for ob in failed:
                    fkey = f"FIN_{ob['key']}"
                    if fkey not in finalized_sent:
                        send_msg(fmt_failed(ob))
                        finalized_sent.add(fkey)
                        logger.info(f"❌ FAILED {ob['type']} {coin_name} {tf_label}")

                # ── 2. Detect new OB from just-closed candle ─
                if closed_candle_too_old(df, tf_label):
                    logger.info(f"   ⏳ {coin_name} {tf_label}: just-closed candle too old")
                else:
                    new_ob = detect_new_ob(df, last_closed_idx, coin_name, tf_label)
                    if new_ob and not overlap_check(new_ob, coin_name, tf_label):
                        fkey = f"FIN_{new_ob['key']}"
                        if (new_ob['key'] not in pending_obs
                                and fkey not in finalized_sent):
                            if new_ob['instant_confirmed']:
                                send_msg(fmt_confirmed(new_ob, instant=True))
                                finalized_sent.add(fkey)
                                logger.info(
                                    f"⚡ INSTANT CONFIRM {new_ob['type']} "
                                    f"{coin_name} {tf_label}")
                            else:
                                with pending_lock:
                                    pending_obs[new_ob['key']] = new_ob
                                logger.info(
                                    f"🟨 OB PENDING {new_ob['type']} "
                                    f"{coin_name} {tf_label}")

                # ── 3. Live pre-alerts ──────────────────────
                triggered = check_live_prealerts(coin_name, tf_label, df)
                for ob in triggered:
                    pre_key = ob['_pre_key']
                    if pre_key not in prealert_sent:
                        send_msg(fmt_pre(ob))
                        prealert_sent.add(pre_key)
                        logger.info(
                            f"🟡 PRE-ALERT {ob['type']} {coin_name} {tf_label}")

                time.sleep(0.2)

        # Trim memory
        if len(prealert_sent) > 5000:
            for s in list(prealert_sent)[:2500]: prealert_sent.discard(s)
        if len(finalized_sent) > 5000:
            for s in list(finalized_sent)[:2500]: finalized_sent.discard(s)

        with pending_lock:
            pend_count = len(pending_obs)
        logger.info(f"✅ Scan complete — {pend_count} pending | "
                    f"{len(prealert_sent)} pre-alerts (lifetime)")
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
            f"body {min(ob['ob_body_top'],ob['ob_body_bot']):.4f}-"
            f"{max(ob['ob_body_top'],ob['ob_body_bot']):.4f}</li>"
            for ob in pending_obs.values()) or '<li>(none)</li>'
    return (
        f"<h2>🤖 OB Alert Bot — Structural + Massive Break</h2>"
        f"<p><b>Status:</b> ✅ Running</p>"
        f"<p><b>Time UTC:</b> {now.strftime('%Y-%m-%d %H:%M:%S')}</p>"
        f"<p><b>Detection:</b> {STREAK_LENGTH}+ streak → "
        f"struct candle ≥ {STRUCT_BODY_MULT}x avg</p>"
        f"<p><b>Confirmation:</b> close beyond OB AND body ≥ {BREAK_BODY_MULT}x avg</p>"
        f"<p><b>Wall-clock fresh:</b> 15m={MAX_CLOSED_CANDLE_AGE_MINUTES['15m']}m, "
        f"1h={MAX_CLOSED_CANDLE_AGE_MINUTES['1h']}m</p>"
        f"<p><b>Pre-alerts (lifetime):</b> {len(prealert_sent)}</p>"
        f"<p><b>Finalized (lifetime):</b> {len(finalized_sent)}</p>"
        f"<h3>Currently Pending:</h3><ul>{pending_html}</ul>"
        f"<h3>Monitoring:</h3><ul>{coins_html}</ul>"
        f"<h3>Timeframes:</h3><ul>{tfs_html}</ul>"
        f"<p><a href='/scan_now'>Force scan</a> | "
        f"<a href='/status'>JSON</a> | <a href='/reset'>Reset</a></p>")


@app.route('/health')
def health(): return 'OK', 200


@app.route('/status')
def status():
    with pending_lock:
        pend = list(pending_obs.values())
    return {
        'status': 'running',
        'mode': 'Structural change + massive break confirm',
        'time_utc': datetime.now(timezone.utc).isoformat(),
        'pending_count': len(pend),
        'pending': [{
            'coin': o['coin'], 'tf': o['tf'], 'type': o['type'],
            'body_top': o['ob_body_top'], 'body_bot': o['ob_body_bot'],
            'wick_high': o['ob_wick_high'], 'wick_low': o['ob_wick_low'],
        } for o in pend],
        'prealerts_lifetime': len(prealert_sent),
        'finalized_lifetime': len(finalized_sent),
        'settings': {
            'streak_length':       STREAK_LENGTH,
            'struct_body_mult':    STRUCT_BODY_MULT,
            'break_body_mult':     BREAK_BODY_MULT,
            'avg_body_period':     AVG_BODY_PERIOD,
            'skip_overlap':        SKIP_OVERLAP,
            'overlap_window':      OVERLAP_WINDOW,
            'wall_clock_fresh':    MAX_CLOSED_CANDLE_AGE_MINUTES,
        },
    }


@app.route('/scan_now')
def scan_now_route():
    threading.Thread(target=scan_all, daemon=True).start()
    return 'Scan triggered!', 200


@app.route('/reset')
def reset():
    with pending_lock:
        pending_obs.clear()
    prealert_sent.clear()
    finalized_sent.clear()
    return 'Reset done!', 200


# ============================================================
# STARTUP
# ============================================================

def start_bot():
    logger.info("OB Bot starting (Structural + Massive Break)…")
    try:
        send_startup_msg()
    except Exception as e:
        logger.error(f"Startup msg error: {e}")
    scheduler = BackgroundScheduler(timezone='UTC')
    scheduler.add_job(scan_all, trigger='cron', minute='*',
        id='ob_scan_job', max_instances=1,
        coalesce=True, misfire_grace_time=20)
    scheduler.start()
    logger.info("Scheduler started — every 1 minute")
    threading.Thread(target=scan_all, daemon=True).start()


threading.Thread(target=start_bot, daemon=True).start()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
