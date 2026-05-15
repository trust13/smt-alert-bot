# ============================================================
# ORDER BLOCK ALERT BOT — LIVE IMPULSE + FVG (v2 with freshness)
# Coins: BTC, ETH, SOL  |  Timeframes: 15m, 1H
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

TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '')
CHAT_ID        = os.environ.get('CHAT_ID', '')

COINS = {'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'SOL': 'SOLUSDT'}
TIMEFRAMES = {'15m': '15m', '1h': '1h'}
BINANCE_US_BASE  = 'https://api.binance.us/api/v3'
BINANCE_INTERVAL = {'15m': '15m', '1h': '1h'}

IMPULSE_MULTIPLIER  = 1.5
AVG_BODY_PERIOD     = 14
OB_SEARCH_LOOKBACK  = 5
REQUIRE_FVG         = True
MAX_OB_TO_FVG_BARS  = 3
SKIP_OVERLAP        = True
OVERLAP_WINDOW      = 20
FRESH_ONLY          = True

# Wall-clock freshness: skip alerts on candles older than this
MAX_CLOSED_CANDLE_AGE_MINUTES = {'15m': 16, '1h': 62}

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

pending_obs    = {}
pending_lock   = threading.Lock()
prealert_sent  = set()
finalized_sent = set()


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


def closed_candle_too_old(df, tf):
    """Returns True if the just-closed candle (n-2) closed too long ago."""
    if df is None or len(df) < 2:
        return True
    close_time = df['close_time'].iloc[-2].to_pydatetime()
    age_min = (datetime.now(timezone.utc) - close_time).total_seconds() / 60
    threshold = MAX_CLOSED_CANDLE_AGE_MINUTES.get(tf, 999999)
    return age_min > threshold


def is_bullish(o, c): return c > o
def is_bearish(o, c): return c < o


def detect_ob_from_candle(df, candle_idx, coin, tf, is_live):
    n = len(df)
    if candle_idx < AVG_BODY_PERIOD or candle_idx + 1 > n:
        return None
    opens, highs, lows, closes, times = (
        df['open'].values, df['high'].values,
        df['low'].values,  df['close'].values, df['time'].values)
    body_window = np.abs(
        closes[candle_idx - AVG_BODY_PERIOD:candle_idx] -
        opens[candle_idx - AVG_BODY_PERIOD:candle_idx])
    if len(body_window) == 0: return None
    avg_body = float(np.mean(body_window))
    if avg_body == 0: return None
    candle_open  = opens[candle_idx]
    candle_close = closes[candle_idx]
    impulse_body = abs(candle_close - candle_open)
    if impulse_body < avg_body * IMPULSE_MULTIPLIER:
        return None
    is_bull_imp = is_bullish(candle_open, candle_close)
    is_bear_imp = is_bearish(candle_open, candle_close)
    if not (is_bull_imp or is_bear_imp): return None
    ob_idx = -1
    search_start = max(0, candle_idx - OB_SEARCH_LOOKBACK)
    for k in range(candle_idx - 1, search_start - 1, -1):
        if is_bull_imp and is_bearish(opens[k], closes[k]):
            ob_idx = k
            break
        if is_bear_imp and is_bullish(opens[k], closes[k]):
            ob_idx = k
            break
    if ob_idx < 0: return None
    if is_bull_imp:
        body_top = opens[ob_idx]; body_bot = closes[ob_idx]
        wick_high = highs[ob_idx]; wick_low = lows[ob_idx]
    else:
        body_top = closes[ob_idx]; body_bot = opens[ob_idx]
        wick_high = highs[ob_idx]; wick_low = lows[ob_idx]
    if REQUIRE_FVG:
        fvg_ok = False
        for offset in range(0, MAX_OB_TO_FVG_BARS + 1):
            c1 = ob_idx + offset
            c3 = c1 + 2
            if c1 + 2 > candle_idx or c3 > candle_idx: break
            if is_bull_imp and lows[c3] > highs[c1]:
                fvg_ok = True; break
            if is_bear_imp and highs[c3] < lows[c1]:
                fvg_ok = True; break
        if not fvg_ok: return None
    if FRESH_ONLY:
        for k in range(ob_idx + 1, candle_idx):
            if is_bull_imp and closes[k] < wick_low: return None
            if is_bear_imp and closes[k] > wick_high: return None
    ob_type = 'BULL' if is_bull_imp else 'BEAR'
    return {
        'key': f"{coin}_{tf}_{ob_type}_{pd.Timestamp(times[ob_idx]).isoformat()}",
        'coin': coin, 'tf': tf, 'type': ob_type,
        'ob_body_top': body_top, 'ob_body_bot': body_bot,
        'ob_wick_high': wick_high, 'ob_wick_low': wick_low,
        'ob_time': times[ob_idx], 'impulse_time': times[candle_idx],
        'is_live': is_live, 'impulse_close': candle_close,
    }


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


def check_pending(coin, tf, df):
    if df is None or len(df) < 2: return [], []
    confirmed, failed = [], []
    last_closed_close = float(df['close'].iloc[-2])
    last_closed_time  = df['time'].iloc[-2]
    with pending_lock:
        keys = [k for k in pending_obs
                if pending_obs[k]['coin']==coin and pending_obs[k]['tf']==tf]
        for key in keys:
            ob = pending_obs[key]
            if ob['impulse_time'] >= last_closed_time: continue
            body_top = max(ob['ob_body_top'], ob['ob_body_bot'])
            body_bot = min(ob['ob_body_top'], ob['ob_body_bot'])
            if ob['type'] == 'BULL':
                if last_closed_close < ob['ob_wick_low']:
                    ob['fail_close'] = last_closed_close
                    ob['fail_time']  = last_closed_time
                    failed.append(ob); del pending_obs[key]
                elif last_closed_close > body_top:
                    ob['confirm_close'] = last_closed_close
                    ob['confirm_time']  = last_closed_time
                    confirmed.append(ob); del pending_obs[key]
            else:
                if last_closed_close > ob['ob_wick_high']:
                    ob['fail_close'] = last_closed_close
                    ob['fail_time']  = last_closed_time
                    failed.append(ob); del pending_obs[key]
                elif last_closed_close < body_bot:
                    ob['confirm_close'] = last_closed_close
                    ob['confirm_time']  = last_closed_time
                    confirmed.append(ob); del pending_obs[key]
    return confirmed, failed


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
            f"Live impulse forming")


def fmt_confirmed(ob, instant=False):
    is_bull = ob['type']=='BULL'
    emoji = '🟢' if is_bull else '🔴'
    label = 'Bull OB' if is_bull else 'Bear OB'
    top = max(ob['ob_body_top'], ob['ob_body_bot'])
    bot_lvl = min(ob['ob_body_top'], ob['ob_body_bot'])
    suffix = " (impulse closed beyond)" if instant else ""
    cls = ob['impulse_close' if instant else 'confirm_close']
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
        f"🤖 <b>OB Alert Bot — LIVE IMPULSE + FVG (v2)</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"<b>Coins:</b> {', '.join(COINS)}\n"
        f"<b>Timeframes:</b> {', '.join(TIMEFRAMES)}\n\n"
        f"<b>Logic:</b>\n"
        f"  1. Live impulsive candle + FVG → 🟡 PRE-ALERT\n"
        f"  2. Impulse closes beyond OB → 🟢/🔴 CONFIRMED instant\n"
        f"  3. Else: pending for next candles\n"
        f"  4. Close opposite past OB wick → ❌ FAILED\n\n"
        f"<b>Wall-clock freshness:</b>\n"
        f"  15m: {MAX_CLOSED_CANDLE_AGE_MINUTES['15m']} min max\n"
        f"  1h:  {MAX_CLOSED_CANDLE_AGE_MINUTES['1h']} min max\n\n"
        f"<i>Data: Binance.US | Scan every 1 min</i>"
    )


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

                live_idx = len(df) - 1
                last_closed_idx = len(df) - 2
                live_candle_time = df['time'].iloc[-1]

                # 1. Check pending
                confirmed, failed = check_pending(coin_name, tf_label, df)
                for ob in confirmed:
                    fkey = f"FIN_{ob['key']}"
                    if fkey not in finalized_sent:
                        send_msg(fmt_confirmed(ob))
                        finalized_sent.add(fkey)
                        logger.info(f"✅ CONFIRMED {ob['type']} {coin_name} {tf_label}")
                for ob in failed:
                    fkey = f"FIN_{ob['key']}"
                    if fkey not in finalized_sent:
                        send_msg(fmt_failed(ob))
                        finalized_sent.add(fkey)
                        logger.info(f"❌ FAILED {ob['type']} {coin_name} {tf_label}")

                # 2. Check just-closed candle (with wall-clock freshness)
                if closed_candle_too_old(df, tf_label):
                    logger.info(f"   ⏳ {coin_name} {tf_label}: closed candle too old, skipping")
                else:
                    closed_ob = detect_ob_from_candle(
                        df, last_closed_idx, coin_name, tf_label, is_live=False)
                    if closed_ob and not overlap_check(closed_ob, coin_name, tf_label):
                        fkey = f"FIN_{closed_ob['key']}"
                        if closed_ob['key'] not in pending_obs and fkey not in finalized_sent:
                            body_top = max(closed_ob['ob_body_top'], closed_ob['ob_body_bot'])
                            body_bot = min(closed_ob['ob_body_top'], closed_ob['ob_body_bot'])
                            imp_close = closed_ob['impulse_close']
                            if closed_ob['type']=='BULL' and imp_close > body_top:
                                send_msg(fmt_confirmed(closed_ob, instant=True))
                                finalized_sent.add(fkey)
                                logger.info(f"🟢 INSTANT CONFIRM Bull {coin_name} {tf_label}")
                            elif closed_ob['type']=='BEAR' and imp_close < body_bot:
                                send_msg(fmt_confirmed(closed_ob, instant=True))
                                finalized_sent.add(fkey)
                                logger.info(f"🔴 INSTANT CONFIRM Bear {coin_name} {tf_label}")
                            else:
                                with pending_lock:
                                    pending_obs[closed_ob['key']] = closed_ob
                                logger.info(f"🟨 OB PENDING {closed_ob['type']} {coin_name} {tf_label}")

                # 3. Live pre-alert
                live_ob = detect_ob_from_candle(df, live_idx, coin_name, tf_label, is_live=True)
                if live_ob and not overlap_check(live_ob, coin_name, tf_label):
                    pre_key = (f"PRE_{coin_name}_{tf_label}_{live_ob['type']}_"
                               f"{pd.Timestamp(live_candle_time).isoformat()}")
                    if pre_key not in prealert_sent:
                        send_msg(fmt_pre(live_ob))
                        prealert_sent.add(pre_key)
                        logger.info(f"🟡 PRE-ALERT {live_ob['type']} {coin_name} {tf_label}")

                time.sleep(0.2)

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
        f"<h2>🤖 OB Alert Bot v2 — Live Impulse + FVG</h2>"
        f"<p><b>Status:</b> ✅ Running</p>"
        f"<p><b>Time UTC:</b> {now.strftime('%Y-%m-%d %H:%M:%S')}</p>"
        f"<p><b>Wall-clock freshness:</b> 15m={MAX_CLOSED_CANDLE_AGE_MINUTES['15m']}m, "
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
        'pending_count': len(pend),
        'prealerts_lifetime': len(prealert_sent),
        'finalized_lifetime': len(finalized_sent),
        'wall_clock_freshness_min': MAX_CLOSED_CANDLE_AGE_MINUTES,
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


def start_bot():
    logger.info("OB Bot starting (v2 with freshness)…")
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
