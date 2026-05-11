# ============================================================
# SMT ALERT BOT — TraderDiegoX Logic Mirror
# Sends Telegram alerts when SMT divergence lines are drawn
# Coins: BTC, ETH, SOL, BNB
# Timeframes: 15m, 1H
# Comparison: NQ1! and YM1! (from stooq.com)
# Crypto data: Binance public API
# ============================================================

import os
import time
import logging
import threading
import requests
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime, timezone, timedelta
from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
import telebot

# ── Config ──────────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '')
CHAT_ID        = os.environ.get('CHAT_ID',        '')

# ── Coins to Monitor (Binance symbols) ──────────────────────
COINS = {
    'BTC':  'BTCUSDT',
    'ETH':  'ETHUSDT',
    'SOL':  'SOLUSDT',
    'BNB':  'BNBUSDT',
}

# ── Comparison Symbols (stooq tickers) ──────────────────────
COMP_SYMBOLS = {
    'NQ1!': 'nq.f',   # Nasdaq 100 Futures continuous
    'YM1!': 'ym.f',   # Dow Jones Futures continuous
}

# ── Timeframes ──────────────────────────────────────────────
TIMEFRAMES = {
    '15m': '15m',
    '1h':  '1h',
}

# ── Binance API ─────────────────────────────────────────────
BINANCE_BASE = 'https://api.binance.com/api/v3'

BINANCE_INTERVAL = {
    '15m': '15m',
    '1h':  '1h',
}

STOOQ_INTERVAL = {
    '15m': '15',
    '1h':  '60',
}

# ── TraderDiegoX Settings (matching script defaults) ────────
PIVOT_LOOKBACK    = 1
PIVOT_A_STRENGTH  = 2
SYNC_TOL          = 2
MAX_SIGNAL_SPAN   = 180
CROSS_TOL_PCT     = 0.02   # 2 * 0.01
MAX_PER_SIDE      = 10
HIDE_INSIDE       = True

# ── Cooldown ────────────────────────────────────────────────
COOLDOWN_SECONDS  = 30 * 60   # 30 minutes per direction per coin per TF

# ── State ───────────────────────────────────────────────────
last_alerts = {}      # key: cooldown_key, value: timestamp
sent_signatures = set()  # unique signature per signal

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

try:
    bot = telebot.TeleBot(TELEGRAM_TOKEN, threaded=False)
    logger.info("Telegram bot initialized")
except Exception as e:
    logger.error(f"Telegram init error: {e}")
    bot = None


# ============================================================
# DATA FETCHING — Binance for crypto, Stooq for futures
# ============================================================

def get_ohlc_binance(symbol, interval):
    """Fetch OHLC from Binance public API (for crypto)"""
    try:
        binance_int = BINANCE_INTERVAL.get(interval, '15m')
        resp = requests.get(
            f"{BINANCE_BASE}/klines",
            params={'symbol': symbol, 'interval': binance_int, 'limit': 500},
            timeout=15
        )
        if resp.status_code != 200:
            logger.error(f"Binance HTTP {resp.status_code} for {symbol}")
            return None
        klines = resp.json()
        if not klines or isinstance(klines, dict):
            return None
        df = pd.DataFrame(klines, columns=[
            'time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'qav', 'num_trades', 'taker_base', 'taker_quote', 'ignore'
        ])
        df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df[['time', 'open', 'high', 'low', 'close']]
    except Exception as e:
        logger.error(f"Binance OHLC error {symbol} {interval}: {e}")
        return None


def get_ohlc_stooq(symbol, interval):
    """Fetch OHLC from stooq.com (for futures like NQ, YM)"""
    try:
        stooq_int = STOOQ_INTERVAL.get(interval, '15')
        url = f"https://stooq.com/q/d/l/?s={symbol}&i={stooq_int}"
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200 or 'No data' in resp.text:
            logger.error(f"Stooq HTTP {resp.status_code} for {symbol}")
            return None
        df = pd.read_csv(StringIO(resp.text))
        if df.empty:
            return None
        df.columns = [c.lower() for c in df.columns]
        if 'date' in df.columns:
            df.rename(columns={'date': 'time'}, inplace=True)
        df['time'] = pd.to_datetime(df['time'], utc=True)
        for col in ['open', 'high', 'low', 'close']:
            df[col] = df[col].astype(float)
        df = df.sort_values('time').reset_index(drop=True)
        return df[['time', 'open', 'high', 'low', 'close']].tail(500)
    except Exception as e:
        logger.error(f"Stooq OHLC error {symbol} {interval}: {e}")
        return None


def get_ohlc(ticker, interval):
    """Smart router: Binance for crypto, stooq for futures"""
    if ticker in COINS.values():
        return get_ohlc_binance(ticker, interval)
    else:
        return get_ohlc_stooq(ticker, interval)


# ============================================================
# PIVOT DETECTION (mirrors ta.pivothigh / ta.pivotlow)
# ============================================================

def detect_pivots(values, left, right):
    """
    Returns (pivots_high, pivots_low) lists of (bar_index, value).
    Matches ta.pivothigh / ta.pivotlow exactly.
    """
    pivots_high = []
    pivots_low  = []
    n = len(values['high'])
    for i in range(left, n - right):
        is_high = True
        is_low  = True
        for k in range(1, left + 1):
            if values['high'][i - k] >= values['high'][i]:
                is_high = False
            if values['low'][i - k] <= values['low'][i]:
                is_low = False
        for k in range(1, right + 1):
            if values['high'][i + k] >= values['high'][i]:
                is_high = False
            if values['low'][i + k] <= values['low'][i]:
                is_low = False
        if is_high:
            pivots_high.append((i, values['high'][i]))
        if is_low:
            pivots_low.append((i, values['low'][i]))
    return pivots_high, pivots_low


# ============================================================
# CROSS TOLERANCE CHECK (matches script's crossOk)
# ============================================================

def cross_ok(highs, lows, bar_a, px_a, bar_b, px_b, is_bear):
    """Check if line between two pivots was crossed by intermediate price"""
    span = bar_b - bar_a - 1
    if span <= 0:
        return True
    if span > MAX_SIGNAL_SPAN:
        return False
    for d in range(1, span + 1):
        idx = bar_a + d
        if idx >= len(highs):
            return False
        line_px = px_a + (px_b - px_a) * (d / (bar_b - bar_a))
        tol_px  = abs(line_px) * CROSS_TOL_PCT / 100.0
        if is_bear:
            if highs[idx] > line_px + tol_px:
                return False
        else:
            if lows[idx] < line_px - tol_px:
                return False
    return True


# ============================================================
# FIND MATCHING PIVOTS IN COMPARISON SYMBOL
# ============================================================

def find_near_pivot(pivots, target_bar, tol):
    """Find pivot in comparison symbol within sync tolerance"""
    best_idx = -1
    best_dist = tol + 1
    for i, (bar, val) in enumerate(pivots):
        dist = abs(bar - target_bar)
        if dist <= tol and dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


def find_near_pivot_before(pivots, target_bar, limit_bar, tol):
    """Find pivot in comparison symbol within tolerance, BEFORE limit"""
    best_idx = -1
    best_dist = tol + 1
    for i, (bar, val) in enumerate(pivots):
        if bar >= limit_bar:
            continue
        dist = abs(bar - target_bar)
        if dist <= tol and dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


# ============================================================
# SMT DETECTION (mirrors TraderDiegoX logic)
# ============================================================

def detect_smt(coin_df, comp_df, comp_label):
    """
    Returns list of SMT signals on the LATEST bar of coin_df.
    """
    if coin_df is None or comp_df is None:
        return []
    if len(coin_df) < 50 or len(comp_df) < 50:
        return []

    # Align lengths
    n = min(len(coin_df), len(comp_df))
    coin_df = coin_df.tail(n).reset_index(drop=True)
    comp_df = comp_df.tail(n).reset_index(drop=True)

    coin_vals = {
        'high': coin_df['high'].values.astype(float),
        'low':  coin_df['low'].values.astype(float),
    }
    comp_vals = {
        'high': comp_df['high'].values.astype(float),
        'low':  comp_df['low'].values.astype(float),
    }

    coin_b_highs, coin_b_lows = detect_pivots(coin_vals, PIVOT_LOOKBACK, PIVOT_LOOKBACK)
    comp_b_highs, comp_b_lows = detect_pivots(comp_vals, PIVOT_LOOKBACK, PIVOT_LOOKBACK)
    coin_a_highs, coin_a_lows = detect_pivots(coin_vals, PIVOT_A_STRENGTH, PIVOT_A_STRENGTH)

    latest_bar = n - 1 - PIVOT_LOOKBACK
    signals = []

    # ── BEARISH SMT (chart leads with new pivot high) ────────
    coin_latest_ph = next(((b, v) for b, v in coin_b_highs if b == latest_bar), None)
    if coin_latest_ph:
        bar_b, px_b = coin_latest_ph
        comp_b_idx = find_near_pivot(comp_b_highs, bar_b, SYNC_TOL)
        if comp_b_idx >= 0:
            comp_bar_b, comp_px_b = comp_b_highs[comp_b_idx]
            for j in range(len(coin_a_highs) - 1, -1, -1):
                bar_a, px_a = coin_a_highs[j]
                span = bar_b - bar_a
                if span <= 0:
                    continue
                if span > MAX_SIGNAL_SPAN:
                    break
                comp_a_idx = find_near_pivot_before(comp_b_highs, bar_a, comp_bar_b, SYNC_TOL)
                if comp_a_idx >= 0:
                    comp_bar_a, comp_px_a = comp_b_highs[comp_a_idx]
                    coin_higher_high = px_b > px_a
                    comp_diverged    = comp_px_b < comp_px_a
                    if coin_higher_high and comp_diverged:
                        if (cross_ok(coin_vals['high'], coin_vals['low'], bar_a, px_a, bar_b, px_b, True) and
                            cross_ok(comp_vals['high'], comp_vals['low'], comp_bar_a, comp_px_a, comp_bar_b, comp_px_b, True)):
                            signals.append({
                                'direction': 'BEAR',
                                'comp_label': comp_label,
                                'coin_bar_a': bar_a, 'coin_px_a': px_a,
                                'coin_bar_b': bar_b, 'coin_px_b': px_b,
                                'comp_bar_a': comp_bar_a, 'comp_px_a': comp_px_a,
                                'comp_bar_b': comp_bar_b, 'comp_px_b': comp_px_b,
                                'span': span,
                                'time_b': coin_df['time'].iloc[bar_b],
                            })
                            break

    # ── BULLISH SMT (chart leads with new pivot low) ─────────
    coin_latest_pl = next(((b, v) for b, v in coin_b_lows if b == latest_bar), None)
    if coin_latest_pl:
        bar_b, px_b = coin_latest_pl
        comp_b_idx = find_near_pivot(comp_b_lows, bar_b, SYNC_TOL)
        if comp_b_idx >= 0:
            comp_bar_b, comp_px_b = comp_b_lows[comp_b_idx]
            for j in range(len(coin_a_lows) - 1, -1, -1):
                bar_a, px_a = coin_a_lows[j]
                span = bar_b - bar_a
                if span <= 0:
                    continue
                if span > MAX_SIGNAL_SPAN:
                    break
                comp_a_idx = find_near_pivot_before(comp_b_lows, bar_a, comp_bar_b, SYNC_TOL)
                if comp_a_idx >= 0:
                    comp_bar_a, comp_px_a = comp_b_lows[comp_a_idx]
                    coin_lower_low = px_b < px_a
                    comp_diverged  = comp_px_b > comp_px_a
                    if coin_lower_low and comp_diverged:
                        if (cross_ok(coin_vals['high'], coin_vals['low'], bar_a, px_a, bar_b, px_b, False) and
                            cross_ok(comp_vals['high'], comp_vals['low'], comp_bar_a, comp_px_a, comp_bar_b, comp_px_b, False)):
                            signals.append({
                                'direction': 'BULL',
                                'comp_label': comp_label,
                                'coin_bar_a': bar_a, 'coin_px_a': px_a,
                                'coin_bar_b': bar_b, 'coin_px_b': px_b,
                                'comp_bar_a': comp_bar_a, 'comp_px_a': comp_px_a,
                                'comp_bar_b': comp_bar_b, 'comp_px_b': comp_px_b,
                                'span': span,
                                'time_b': coin_df['time'].iloc[bar_b],
                            })
                            break

    return signals


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


def format_signal(coin, tf, signal):
    direction = signal['direction']
    emoji     = '🟢' if direction == 'BULL' else '🔴'
    arrow_coin = '↓ Lower Low' if direction == 'BULL' else '↑ Higher High'
    arrow_comp = '↑ Higher Low (DIVERGED)' if direction == 'BULL' else '↓ Lower High (DIVERGED)'
    watch     = '🎯 Watch for reversal UP'   if direction == 'BULL' else '🎯 Watch for reversal DOWN'

    time_str = pd.Timestamp(signal['time_b']).strftime('%H:%M UTC %d-%b')

    msg = (
        f"{emoji} <b>{direction} SMT — {coin}/USDT {tf}</b>\n"
        f"{'─'*30}\n"
        f"<b>Coin Pivot A:</b> ${signal['coin_px_a']:,.4f} ({signal['span']} bars ago)\n"
        f"<b>Coin Pivot B:</b> ${signal['coin_px_b']:,.4f} (now)\n"
        f"<b>Direction:</b> {arrow_coin}\n\n"
        f"<b>Comparison:</b> {signal['comp_label']}\n"
        f"<b>Comp Pivot A:</b> {signal['comp_px_a']:,.2f}\n"
        f"<b>Comp Pivot B:</b> {signal['comp_px_b']:,.2f}\n"
        f"<b>Direction:</b> {arrow_comp}\n\n"
        f"<b>Span:</b> {signal['span']} bars\n"
        f"<b>Time:</b> {time_str}\n\n"
        f"{watch}"
    )
    return msg


def send_startup_msg():
    coins_str = ', '.join(COINS.keys())
    comps_str = ', '.join(COMP_SYMBOLS.keys())
    tfs_str   = ', '.join(TIMEFRAMES.keys())
    msg = (
        f"🤖 <b>SMT Alert Bot — LIVE</b>\n"
        f"{'─'*30}\n"
        f"<b>Coins:</b> {coins_str}\n"
        f"<b>Timeframes:</b> {tfs_str}\n"
        f"<b>Comparisons:</b> {comps_str}\n\n"
        f"<b>Data Sources:</b>\n"
        f"  Crypto: Binance API\n"
        f"  Futures: stooq.com\n\n"
        f"<b>Settings:</b>\n"
        f"  Pivot Lookback: {PIVOT_LOOKBACK}\n"
        f"  Pivot A Strength: {PIVOT_A_STRENGTH}\n"
        f"  Sync Tolerance: {SYNC_TOL}\n"
        f"  Max Span: {MAX_SIGNAL_SPAN}\n"
        f"  Cross Tol: {CROSS_TOL_PCT}%\n"
        f"  Cooldown: 30 min\n\n"
        f"<i>Scanning every minute…</i>"
    )
    send_msg(msg)


# ============================================================
# CORE SCAN
# ============================================================

def scan_coin_tf(coin_name, coin_ticker, tf_label, tf_interval):
    try:
        coin_df = get_ohlc(coin_ticker, tf_interval)
        if coin_df is None:
            return

        for comp_label, comp_ticker in COMP_SYMBOLS.items():
            comp_df = get_ohlc(comp_ticker, tf_interval)
            if comp_df is None:
                continue

            signals = detect_smt(coin_df, comp_df, comp_label)
            for sig in signals:
                signature = (
                    f"{coin_name}_{tf_label}_{sig['direction']}_"
                    f"{sig['comp_label']}_"
                    f"{int(sig['coin_bar_a'])}_{int(sig['coin_bar_b'])}_"
                    f"{round(sig['coin_px_a'], 4)}_{round(sig['coin_px_b'], 4)}"
                )
                if signature in sent_signatures:
                    continue

                cooldown_key = f"{coin_name}_{tf_label}_{sig['direction']}_{sig['comp_label']}"
                last_time = last_alerts.get(cooldown_key, 0)
                if time.time() - last_time < COOLDOWN_SECONDS:
                    continue

                msg = format_signal(coin_name, tf_label, sig)
                send_msg(msg)
                sent_signatures.add(signature)
                last_alerts[cooldown_key] = time.time()
                logger.info(
                    f"✅ {sig['direction']} SMT {coin_name} {tf_label} vs {sig['comp_label']} "
                    f"@ {sig['coin_px_b']:.4f}"
                )

        if len(sent_signatures) > 2000:
            for s in list(sent_signatures)[:1000]:
                sent_signatures.discard(s)

    except Exception as e:
        logger.error(f"scan_coin_tf {coin_name} {tf_label}: {e}")


def scan_all():
    try:
        now = datetime.now(timezone.utc)
        logger.info(f"🔍 Scan cycle started at {now.strftime('%H:%M:%S UTC')}")
        for coin_name, coin_ticker in COINS.items():
            for tf_label, tf_interval in TIMEFRAMES.items():
                scan_coin_tf(coin_name, coin_ticker, tf_label, tf_interval)
                time.sleep(0.5)
        logger.info(f"✅ Scan cycle complete. Sent: {len(sent_signatures)} total")
    except Exception as e:
        logger.error(f"scan_all error: {e}")


# ============================================================
# FLASK ROUTES
# ============================================================

@app.route('/')
def index():
    now = datetime.now(timezone.utc)
    coins_html = ''.join([f"<li>{c}</li>" for c in COINS.keys()])
    return (
        f"<h2>🤖 SMT Alert Bot</h2>"
        f"<p><b>Status:</b> ✅ Running</p>"
        f"<p><b>Time UTC:</b> {now.strftime('%Y-%m-%d %H:%M:%S')}</p>"
        f"<p><b>Comparisons:</b> NQ1!, YM1!</p>"
        f"<p><b>Timeframes:</b> 15m, 1H</p>"
        f"<p><b>Crypto data:</b> Binance API</p>"
        f"<p><b>Futures data:</b> stooq.com</p>"
        f"<p><b>Total alerts sent:</b> {len(sent_signatures)}</p>"
        f"<p><b>Active cooldowns:</b> {len(last_alerts)}</p>"
        f"<h3>Monitoring:</h3><ul>{coins_html}</ul>"
    )


@app.route('/health')
def health():
    return 'OK', 200


@app.route('/status')
def status():
    return {
        'status':         'running',
        'time_utc':       datetime.now(timezone.utc).isoformat(),
        'coins':          list(COINS.keys()),
        'timeframes':     list(TIMEFRAMES.keys()),
        'comparisons':    list(COMP_SYMBOLS.keys()),
        'alerts_sent':    len(sent_signatures),
        'cooldowns':      len(last_alerts),
    }


@app.route('/scan_now')
def scan_now():
    threading.Thread(target=scan_all, daemon=True).start()
    return 'Scan triggered!', 200


@app.route('/reset')
def reset():
    sent_signatures.clear()
    last_alerts.clear()
    return 'Reset done!', 200


# ============================================================
# STARTUP
# ============================================================

def start_bot():
    logger.info("SMT Bot starting…")
    try:
        send_startup_msg()
    except Exception as e:
        logger.error(f"Startup msg error: {e}")

    scheduler = BackgroundScheduler(timezone='UTC')
    scheduler.add_job(
        scan_all, trigger='cron', minute='*',
        id='scan_job', max_instances=1,
        coalesce=True, misfire_grace_time=30
    )
    scheduler.start()
    logger.info("Scheduler started")
    threading.Thread(target=scan_all, daemon=True).start()


threading.Thread(target=start_bot, daemon=True).start()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
