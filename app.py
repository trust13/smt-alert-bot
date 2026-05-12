# ============================================================
# SMT ALERT BOT — TraderDiegoX Logic Mirror
# Sends Telegram alerts when SMT divergence lines are drawn
# Coins: BTC, ETH, SOL, BNB
# Timeframes: 15m, 1H
# Comparison: BTC Dominance (BTC.D)
# Crypto data: Binance.US public API (works from Render)
# BTC.D data: CoinGecko free API (no key needed)
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

# ── Config ──────────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '')
CHAT_ID        = os.environ.get('CHAT_ID',        '')

# ── Coins to Monitor (Binance.US symbols) ───────────────────
COINS = {
    'BTC': 'BTCUSDT',
    'ETH': 'ETHUSDT',
    'SOL': 'SOLUSDT',
    'BNB': 'BNBUSDT',
}

# ── Comparison Symbol ────────────────────────────────────────
COMP_LABEL  = 'BTC.D'

# ── Timeframes ──────────────────────────────────────────────
TIMEFRAMES = {
    '15m': '15m',
    '1h':  '1h',
}

# ── API Endpoints ───────────────────────────────────────────
BINANCE_US_BASE  = 'https://api.binance.us/api/v3'
COINGECKO_BASE   = 'https://api.coingecko.com/api/v3'

BINANCE_INTERVAL = {
    '15m': '15m',
    '1h':  '1h',
}

# CoinGecko market chart days needed per timeframe
# 15m → need last 2 days (gives 5-min granularity, we resample)
# 1h  → need last 90 days (gives hourly granularity)
COINGECKO_DAYS = {
    '15m': 2,
    '1h':  90,
}

# ── TraderDiegoX Settings ────────────────────────────────────
PIVOT_LOOKBACK   = 1
PIVOT_A_STRENGTH = 2
SYNC_TOL         = 2
MAX_SIGNAL_SPAN  = 180
CROSS_TOL_PCT    = 0.02
HIDE_INSIDE      = True

# ── Cooldown ─────────────────────────────────────────────────
COOLDOWN_SECONDS = 30 * 60

# ── State ────────────────────────────────────────────────────
last_alerts      = {}
sent_signatures  = set()

# Cache BTC.D per timeframe (refresh every 90 seconds to respect CoinGecko rate limits)
btcd_cache      = {}
btcd_cache_time = {}
BTCD_CACHE_TTL  = 90  # seconds

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
# BTC DOMINANCE — CoinGecko
# ============================================================

def fetch_btc_dominance_ohlc(interval):
    """
    Build BTC Dominance OHLC candles from CoinGecko.

    Method:
      - Fetch BTC market cap history  (market_chart)
      - Fetch Total market cap history (global/market_cap_chart)
      - Compute BTC.D = btc_mcap / total_mcap * 100
      - Resample into requested interval candles (OHLC)

    CoinGecko free tier:
      - /coins/{id}/market_chart  → market caps, prices, volumes
      - /global/market_cap_chart  → total market cap history
      - Rate limit: 10-30 calls/min depending on server load
    """
    cache_key = f"btcd_{interval}"
    now_ts = time.time()

    if cache_key in btcd_cache and cache_key in btcd_cache_time:
        if now_ts - btcd_cache_time[cache_key] < BTCD_CACHE_TTL:
            return btcd_cache[cache_key]

    try:
        days = COINGECKO_DAYS.get(interval, 2)

        # ── Step 1: BTC market cap history ──────────────────
        btc_resp = requests.get(
            f"{COINGECKO_BASE}/coins/bitcoin/market_chart",
            params={
                'vs_currency': 'usd',
                'days':        days,
                'interval':    'hourly' if interval == '1h' else 'minutely',
            },
            timeout=20,
            headers={'Accept': 'application/json'}
        )
        if btc_resp.status_code == 429:
            logger.warning("CoinGecko rate limited — using cache if available")
            return btcd_cache.get(cache_key)
        if btc_resp.status_code != 200:
            logger.error(f"CoinGecko BTC market_chart HTTP {btc_resp.status_code}")
            return btcd_cache.get(cache_key)

        btc_data   = btc_resp.json()
        btc_mcaps  = btc_data.get('market_caps', [])
        if not btc_mcaps:
            logger.error("CoinGecko BTC market_caps empty")
            return btcd_cache.get(cache_key)

        btc_df = pd.DataFrame(btc_mcaps, columns=['ts_ms', 'btc_mcap'])
        btc_df['time'] = pd.to_datetime(btc_df['ts_ms'], unit='ms', utc=True)
        btc_df = btc_df[['time', 'btc_mcap']].set_index('time')

        # Small pause to avoid hitting rate limit
        time.sleep(1.5)

        # ── Step 2: Total market cap history ────────────────
        total_resp = requests.get(
            f"{COINGECKO_BASE}/global/market_cap_chart",
            params={'days': days},
            timeout=20,
            headers={'Accept': 'application/json'}
        )
        if total_resp.status_code == 429:
            logger.warning("CoinGecko rate limited on total mcap — using cache")
            return btcd_cache.get(cache_key)
        if total_resp.status_code != 200:
            logger.error(f"CoinGecko total market_cap_chart HTTP {total_resp.status_code}")
            return btcd_cache.get(cache_key)

        total_data  = total_resp.json()
        total_mcaps = total_data.get('market_cap_chart', {}).get('market_cap', [])
        if not total_mcaps:
            logger.error("CoinGecko total market_cap empty")
            return btcd_cache.get(cache_key)

        total_df = pd.DataFrame(total_mcaps, columns=['ts_ms', 'total_mcap'])
        total_df['time'] = pd.to_datetime(total_df['ts_ms'], unit='ms', utc=True)
        total_df = total_df[['time', 'total_mcap']].set_index('time')

        # ── Step 3: Merge and compute BTC.D ─────────────────
        merged = btc_df.join(total_df, how='inner')
        if len(merged) < 10:
            # Try outer join with forward fill
            merged = btc_df.join(total_df, how='outer').sort_index().ffill().dropna()
        if len(merged) < 10:
            logger.error("BTC.D merged data too short")
            return btcd_cache.get(cache_key)

        merged['btcd'] = (merged['btc_mcap'] / merged['total_mcap']) * 100.0
        merged = merged[['btcd']].copy()

        # ── Step 4: Resample into OHLC candles ──────────────
        resample_rule = {
            '15m': '15min',
            '1h':  '1h',
        }.get(interval, '15min')

        ohlc = merged['btcd'].resample(resample_rule).ohlc().dropna()
        ohlc = ohlc.reset_index()
        ohlc.columns = ['time', 'open', 'high', 'low', 'close']
        ohlc = ohlc.sort_values('time').reset_index(drop=True)

        if len(ohlc) < 20:
            logger.error(f"BTC.D OHLC too few candles: {len(ohlc)}")
            return btcd_cache.get(cache_key)

        logger.info(
            f"BTC.D {interval}: {len(ohlc)} candles, "
            f"latest={ohlc['close'].iloc[-1]:.2f}%"
        )

        btcd_cache[cache_key]      = ohlc
        btcd_cache_time[cache_key] = now_ts
        return ohlc

    except Exception as e:
        logger.error(f"fetch_btc_dominance_ohlc {interval}: {e}")
        return btcd_cache.get(cache_key)


# ============================================================
# DATA FETCHING — Binance.US for crypto
# ============================================================

def get_ohlc_binance(symbol, interval):
    """Fetch OHLC from Binance.US"""
    try:
        binance_int = BINANCE_INTERVAL.get(interval, '15m')
        resp = requests.get(
            f"{BINANCE_US_BASE}/klines",
            params={
                'symbol':   symbol,
                'interval': binance_int,
                'limit':    500,
            },
            timeout=15
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
            'taker_base', 'taker_quote', 'ignore'
        ])
        df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        return df[['time', 'open', 'high', 'low', 'close']]

    except Exception as e:
        logger.error(f"Binance.US OHLC error {symbol} {interval}: {e}")
        return None


# ============================================================
# PIVOT DETECTION
# ============================================================

def detect_pivots(values, left, right):
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
                is_low  = False
        for k in range(1, right + 1):
            if values['high'][i + k] >= values['high'][i]:
                is_high = False
            if values['low'][i + k] <= values['low'][i]:
                is_low  = False
        if is_high:
            pivots_high.append((i, values['high'][i]))
        if is_low:
            pivots_low.append((i, values['low'][i]))
    return pivots_high, pivots_low


def cross_ok(highs, lows, bar_a, px_a, bar_b, px_b, is_bear):
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


def find_near_pivot(pivots, target_bar, tol):
    best_idx = -1
    best_dist = tol + 1
    for i, (bar, val) in enumerate(pivots):
        dist = abs(bar - target_bar)
        if dist <= tol and dist < best_dist:
            best_dist = dist
            best_idx  = i
    return best_idx


def find_near_pivot_before(pivots, target_bar, limit_bar, tol):
    best_idx = -1
    best_dist = tol + 1
    for i, (bar, val) in enumerate(pivots):
        if bar >= limit_bar:
            continue
        dist = abs(bar - target_bar)
        if dist <= tol and dist < best_dist:
            best_dist = dist
            best_idx  = i
    return best_idx


# ============================================================
# SMT DETECTION
# ============================================================

def detect_smt(coin_df, comp_df, comp_label):
    """
    Detect SMT divergence between coin OHLC and comparison OHLC.
    Works identically whether comp is NQ, YM, or BTC.D —
    just OHLC DataFrames with [time, open, high, low, close].
    """
    if coin_df is None or comp_df is None:
        return []
    if len(coin_df) < 50 or len(comp_df) < 50:
        return []

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

    coin_b_highs, coin_b_lows = detect_pivots(
        coin_vals, PIVOT_LOOKBACK, PIVOT_LOOKBACK
    )
    comp_b_highs, comp_b_lows = detect_pivots(
        comp_vals, PIVOT_LOOKBACK, PIVOT_LOOKBACK
    )
    coin_a_highs, coin_a_lows = detect_pivots(
        coin_vals, PIVOT_A_STRENGTH, PIVOT_A_STRENGTH
    )

    latest_bar = n - 1 - PIVOT_LOOKBACK
    signals    = []

    # ── BEAR: coin makes Higher High, BTC.D makes Lower High ──
    coin_latest_ph = next(
        ((b, v) for b, v in coin_b_highs if b == latest_bar), None
    )
    if coin_latest_ph:
        bar_b, px_b = coin_latest_ph
        comp_b_idx  = find_near_pivot(comp_b_highs, bar_b, SYNC_TOL)
        if comp_b_idx >= 0:
            comp_bar_b, comp_px_b = comp_b_highs[comp_b_idx]
            for j in range(len(coin_a_highs) - 1, -1, -1):
                bar_a, px_a = coin_a_highs[j]
                span = bar_b - bar_a
                if span <= 0:
                    continue
                if span > MAX_SIGNAL_SPAN:
                    break
                comp_a_idx = find_near_pivot_before(
                    comp_b_highs, bar_a, comp_bar_b, SYNC_TOL
                )
                if comp_a_idx >= 0:
                    comp_bar_a, comp_px_a = comp_b_highs[comp_a_idx]
                    if px_b > px_a and comp_px_b < comp_px_a:
                        if (cross_ok(coin_vals['high'], coin_vals['low'],
                                     bar_a, px_a, bar_b, px_b, True) and
                            cross_ok(comp_vals['high'], comp_vals['low'],
                                     comp_bar_a, comp_px_a,
                                     comp_bar_b, comp_px_b, True)):
                            signals.append({
                                'direction':  'BEAR',
                                'comp_label': comp_label,
                                'coin_bar_a': bar_a,   'coin_px_a': px_a,
                                'coin_bar_b': bar_b,   'coin_px_b': px_b,
                                'comp_bar_a': comp_bar_a, 'comp_px_a': comp_px_a,
                                'comp_bar_b': comp_bar_b, 'comp_px_b': comp_px_b,
                                'span':       span,
                                'time_b':     coin_df['time'].iloc[bar_b],
                            })
                            break

    # ── BULL: coin makes Lower Low, BTC.D makes Higher Low ────
    coin_latest_pl = next(
        ((b, v) for b, v in coin_b_lows if b == latest_bar), None
    )
    if coin_latest_pl:
        bar_b, px_b = coin_latest_pl
        comp_b_idx  = find_near_pivot(comp_b_lows, bar_b, SYNC_TOL)
        if comp_b_idx >= 0:
            comp_bar_b, comp_px_b = comp_b_lows[comp_b_idx]
            for j in range(len(coin_a_lows) - 1, -1, -1):
                bar_a, px_a = coin_a_lows[j]
                span = bar_b - bar_a
                if span <= 0:
                    continue
                if span > MAX_SIGNAL_SPAN:
                    break
                comp_a_idx = find_near_pivot_before(
                    comp_b_lows, bar_a, comp_bar_b, SYNC_TOL
                )
                if comp_a_idx >= 0:
                    comp_bar_a, comp_px_a = comp_b_lows[comp_a_idx]
                    if px_b < px_a and comp_px_b > comp_px_a:
                        if (cross_ok(coin_vals['high'], coin_vals['low'],
                                     bar_a, px_a, bar_b, px_b, False) and
                            cross_ok(comp_vals['high'], comp_vals['low'],
                                     comp_bar_a, comp_px_a,
                                     comp_bar_b, comp_px_b, False)):
                            signals.append({
                                'direction':  'BULL',
                                'comp_label': comp_label,
                                'coin_bar_a': bar_a,   'coin_px_a': px_a,
                                'coin_bar_b': bar_b,   'coin_px_b': px_b,
                                'comp_bar_a': comp_bar_a, 'comp_px_a': comp_px_a,
                                'comp_bar_b': comp_bar_b, 'comp_px_b': comp_px_b,
                                'span':       span,
                                'time_b':     coin_df['time'].iloc[bar_b],
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
    direction  = signal['direction']
    emoji      = '🟢' if direction == 'BULL' else '🔴'

    if direction == 'BULL':
        arrow_coin = '↓ Lower Low'
        arrow_comp = '↑ Higher Low  ← BTC.D DIVERGED'
        watch      = '🎯 Watch for reversal UP — altcoin strength vs BTC dominance'
        interp     = (
            'BTC Dominance is rising (BTC stronger) while coin '
            'makes a lower low — classic SMT BULL setup.'
        )
    else:
        arrow_coin = '↑ Higher High'
        arrow_comp = '↓ Lower High  ← BTC.D DIVERGED'
        watch      = '🎯 Watch for reversal DOWN — altcoin weakness vs BTC dominance'
        interp     = (
            'BTC Dominance is falling (alts stronger) while coin '
            'makes a higher high — classic SMT BEAR setup.'
        )

    time_str = pd.Timestamp(signal['time_b']).strftime('%H:%M UTC %d-%b')

    msg = (
        f"{emoji} <b>{direction} SMT — {coin}/USDT {tf}</b>\n"
        f"{'─'*32}\n"
        f"<b>Coin Pivot A:</b>  ${signal['coin_px_a']:,.4f}  ({signal['span']} bars ago)\n"
        f"<b>Coin Pivot B:</b>  ${signal['coin_px_b']:,.4f}  (now)\n"
        f"<b>Coin Move:</b>     {arrow_coin}\n\n"
        f"<b>Comparison:</b>   {signal['comp_label']}\n"
        f"<b>BTC.D Pivot A:</b> {signal['comp_px_a']:.3f}%\n"
        f"<b>BTC.D Pivot B:</b> {signal['comp_px_b']:.3f}%\n"
        f"<b>BTC.D Move:</b>   {arrow_comp}\n\n"
        f"<b>Span:</b>  {signal['span']} bars\n"
        f"<b>Time:</b>  {time_str}\n\n"
        f"<i>{interp}</i>\n\n"
        f"{watch}"
    )
    return msg


def send_startup_msg():
    coins_str = ', '.join(COINS.keys())
    tfs_str   = ', '.join(TIMEFRAMES.keys())
    msg = (
        f"🤖 <b>SMT Alert Bot — LIVE</b>\n"
        f"{'─'*32}\n"
        f"<b>Coins:</b>       {coins_str}\n"
        f"<b>Timeframes:</b>  {tfs_str}\n"
        f"<b>Comparison:</b>  BTC Dominance (BTC.D)\n\n"
        f"<b>Data Sources:</b>\n"
        f"  Crypto:  Binance.US API  (live, no key)\n"
        f"  BTC.D:   CoinGecko API   (live, no key)\n\n"
        f"<b>SMT Logic:</b>\n"
        f"  🟢 BULL = Coin Lower Low + BTC.D Higher Low\n"
        f"  🔴 BEAR = Coin Higher High + BTC.D Lower High\n\n"
        f"<b>Settings:</b>\n"
        f"  Pivot Lookback:  {PIVOT_LOOKBACK}\n"
        f"  Pivot A Strength:{PIVOT_A_STRENGTH}\n"
        f"  Sync Tolerance:  {SYNC_TOL}\n"
        f"  Max Span:        {MAX_SIGNAL_SPAN}\n"
        f"  Cooldown:        30 min\n\n"
        f"<i>Scanning every minute…</i>"
    )
    send_msg(msg)


# ============================================================
# CORE SCAN
# ============================================================

def scan_coin_tf(coin_name, coin_ticker, tf_label, tf_interval):
    try:
        coin_df = get_ohlc_binance(coin_ticker, tf_interval)
        if coin_df is None:
            logger.warning(f"No coin data: {coin_name} {tf_label}")
            return

        comp_df = fetch_btc_dominance_ohlc(tf_interval)
        if comp_df is None:
            logger.warning(f"No BTC.D data for {tf_label}")
            return

        signals = detect_smt(coin_df, comp_df, COMP_LABEL)

        for sig in signals:
            signature = (
                f"{coin_name}_{tf_label}_{sig['direction']}_"
                f"BTCD_"
                f"{int(sig['coin_bar_a'])}_{int(sig['coin_bar_b'])}_"
                f"{round(sig['coin_px_a'], 4)}_{round(sig['coin_px_b'], 4)}"
            )
            if signature in sent_signatures:
                continue

            cooldown_key = f"{coin_name}_{tf_label}_{sig['direction']}_BTCD"
            last_time    = last_alerts.get(cooldown_key, 0)
            if time.time() - last_time < COOLDOWN_SECONDS:
                continue

            msg = format_signal(coin_name, tf_label, sig)
            send_msg(msg)
            sent_signatures.add(signature)
            last_alerts[cooldown_key] = time.time()
            logger.info(
                f"✅ {sig['direction']} SMT {coin_name} {tf_label} vs BTC.D "
                f"@ coin={sig['coin_px_b']:.4f} btcd={sig['comp_px_b']:.3f}%"
            )

        # Cleanup old signatures
        if len(sent_signatures) > 2000:
            for s in list(sent_signatures)[:1000]:
                sent_signatures.discard(s)

    except Exception as e:
        logger.error(f"scan_coin_tf {coin_name} {tf_label}: {e}")


def scan_all():
    try:
        now = datetime.now(timezone.utc)
        logger.info(f"🔍 Scan cycle at {now.strftime('%H:%M:%S UTC')}")

        for coin_name, coin_ticker in COINS.items():
            for tf_label, tf_interval in TIMEFRAMES.items():
                scan_coin_tf(coin_name, coin_ticker, tf_label, tf_interval)
                time.sleep(1)  # Gentle pacing between calls

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
    now = datetime.now(timezone.utc)
    coins_html = ''.join([f"<li>{c}/USDT</li>" for c in COINS.keys()])
    btcd_status = {}
    for tf in TIMEFRAMES:
        df = btcd_cache.get(f"btcd_{tf}")
        if df is not None:
            btcd_status[tf] = (
                f"{len(df)} candles, "
                f"latest={df['close'].iloc[-1]:.3f}%"
            )
        else:
            btcd_status[tf] = "not yet fetched"

    btcd_html = ''.join([
        f"<li>{tf}: {v}</li>" for tf, v in btcd_status.items()
    ])

    return (
        f"<h2>🤖 SMT Alert Bot</h2>"
        f"<p><b>Status:</b> ✅ Running</p>"
        f"<p><b>Time UTC:</b> {now.strftime('%Y-%m-%d %H:%M:%S')}</p>"
        f"<p><b>Comparison:</b> BTC Dominance (BTC.D)</p>"
        f"<p><b>Timeframes:</b> 15m, 1H</p>"
        f"<p><b>Crypto data:</b> Binance.US API (live)</p>"
        f"<p><b>BTC.D data:</b> CoinGecko API (live)</p>"
        f"<p><b>Total alerts sent:</b> {len(sent_signatures)}</p>"
        f"<p><b>Active cooldowns:</b> {len(last_alerts)}</p>"
        f"<h3>Monitoring:</h3><ul>{coins_html}</ul>"
        f"<h3>BTC.D Cache:</h3><ul>{btcd_html}</ul>"
    )


@app.route('/health')
def health():
    return 'OK', 200


@app.route('/status')
def status():
    btcd_info = {}
    for tf in TIMEFRAMES:
        df = btcd_cache.get(f"btcd_{tf}")
        btcd_info[tf] = {
            'candles': len(df) if df is not None else 0,
            'latest':  float(df['close'].iloc[-1]) if df is not None else None,
        }
    return {
        'status':       'running',
        'time_utc':     datetime.now(timezone.utc).isoformat(),
        'coins':        list(COINS.keys()),
        'timeframes':   list(TIMEFRAMES.keys()),
        'comparison':   'BTC.D',
        'alerts_sent':  len(sent_signatures),
        'cooldowns':    len(last_alerts),
        'btcd_cache':   btcd_info,
    }


@app.route('/scan_now')
def scan_now():
    threading.Thread(target=scan_all, daemon=True).start()
    return 'Scan triggered!', 200


@app.route('/reset')
def reset():
    sent_signatures.clear()
    last_alerts.clear()
    btcd_cache.clear()
    btcd_cache_time.clear()
    return 'Reset done!', 200


@app.route('/btcd')
def show_btcd():
    """Debug endpoint: show current BTC.D cache"""
    lines = []
    for tf in TIMEFRAMES:
        df = btcd_cache.get(f"btcd_{tf}")
        if df is not None:
            tail = df.tail(5)
            lines.append(f"<h3>BTC.D {tf} (last 5 candles)</h3><pre>")
            lines.append(tail.to_string())
            lines.append("</pre>")
        else:
            lines.append(f"<h3>BTC.D {tf}</h3><p>No data yet</p>")
    return ''.join(lines)


# ============================================================
# STARTUP
# ============================================================

def start_bot():
    logger.info("SMT Bot starting — BTC Dominance mode")
    try:
        send_startup_msg()
    except Exception as e:
        logger.error(f"Startup msg error: {e}")

    # Pre-warm BTC.D cache
    logger.info("Pre-warming BTC.D cache…")
    for tf in TIMEFRAMES:
        df = fetch_btc_dominance_ohlc(tf)
        if df is not None:
            logger.info(f"BTC.D {tf}: {len(df)} candles loaded")
        else:
            logger.warning(f"BTC.D {tf}: failed to load")
        time.sleep(2)

    scheduler = BackgroundScheduler(timezone='UTC')
    scheduler.add_job(
        scan_all,
        trigger='cron',
        minute='*',
        id='scan_job',
        max_instances=1,
        coalesce=True,
        misfire_grace_time=30
    )
    scheduler.start()
    logger.info("Scheduler started — scanning every minute")
    threading.Thread(target=scan_all, daemon=True).start()


threading.Thread(target=start_bot, daemon=True).start()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
