import pandas as pd
import numpy as np
import datetime
import time
import json
import os
import requests
import joblib
import sys
import websocket
import threading
import asyncio
from db.database import init_db, insert_trade_log, insert_placed_trade

# Add script dir
sys.path.append(os.path.dirname(__file__))

# --- Configuration ---
LIGHTER_WS_URL = "wss://mainnet.zklighter.elliot.ai/stream"
LIGHTER_REST_URL = "https://mainnet.zklighter.elliot.ai/api/v1"

MARKET_ID = 1
SYMBOL = "BTC"
TIMEFRAME = "5m"
TIMEFRAME_MINUTES = 5

# Trading Params (V4.1 Moto Shield V2)
INITIAL_CAPITAL = 300.0
LEVERAGE = 30
CONFIDENCE_THRESHOLD = 0.65

# MOTO 1 PARAMS (LOSS SHIELD)
MAX_SMA_DIST = 0.015  # Tightened from 0.02
MAX_RSI_LONG = 75
MIN_RSI_SHORT = 25

# MOTO 2 PARAMS (TREND SWORD) - DISABLED for now as it lost money
ADX_STRONG = 100 # Effectively Disabled
MFI_RELAXED_LONG = -25 
MFI_RELAXED_SHORT = 25

# --- Paths ---
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "trade_classifier_v4_1.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_v4_1.pkl")

# --- Globals ---
model = None
scaler = None
candles_df = pd.DataFrame()

# State
current_balance = INITIAL_CAPITAL
active_trade = None

# --- ML Loader ---
def load_ml_model():
    global model, scaler
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("ML Model V4.1 (Base) Loaded Successfully.")
    except Exception as e:
        print(f"Failed to load ML model V4.1: {e}")
        model = None

# --- Indicator Logic (V4.1 Logic) ---
def full_indicator_calc(df):
    high = df['high']; low = df['low']; close = df['close']; vol = df['volume']
    
    # Smart Trail (V4.1 Logic)
    high_prev = high.shift(1); low_prev = low.shift(1); close_prev = close.shift(1)
    range_hl = high - low
    sma_range = range_hl.rolling(window=200).mean().fillna(0)
    hilo = np.minimum(range_hl, 1.5 * sma_range)
    href = np.where(low <= high_prev, high - close_prev, (high - close_prev) - 0.5 * (low - high_prev))
    lref = np.where(high >= low_prev, close_prev - low, (close_prev - low) - 0.5 * (low_prev - high))
    true_range = np.maximum(hilo, np.maximum(href, lref))
    loss = 4.2 * pd.Series(true_range, index=df.index).ewm(alpha=1/200, adjust=False).mean()
    up68 = close - loss; dn68 = close + loss
    
    t_up = np.zeros(len(df)); t_dn = np.zeros(len(df)); tr = np.zeros(len(df), dtype=int)
    close_vals = close.values; up_vals = up68.values; dn_vals = dn68.values
    if len(df)>0: t_up[0],t_dn[0],tr[0]=up_vals[0],dn_vals[0],1
    for i in range(1, len(df)):
        c_prev, c_curr = close_vals[i-1], close_vals[i]
        up_curr, dn_curr = up_vals[i], dn_vals[i]
        t_up_prev, t_dn_prev = t_up[i-1], t_dn[i-1]
        t_up[i] = max(up_curr, t_up_prev) if c_prev > t_up_prev else up_curr
        t_dn[i] = min(dn_curr, t_dn_prev) if c_prev < t_dn_prev else dn_curr
        if c_curr > t_dn[i-1]: tr[i] = 1
        elif c_curr < t_up[i-1]: tr[i] = -1
        else: tr[i] = tr[i-1]
    df['SmartTrail_Trend'] = tr
    
    # Lux Matrix
    L = 7
    hl2 = (high + low) / 2
    hw_hi = high.rolling(window=L).max(); hw_lo = low.rolling(window=L).min()
    hw_av = hl2.rolling(window=L).mean()
    norm_src = (close - (hw_hi + hw_lo + hw_av) / 3) / (hw_hi - hw_lo) * 100
    norm_src = norm_src.replace([np.inf, -np.inf], 0).fillna(0)
    x = np.arange(L); mean_x = np.mean(x); var_x = np.var(x)
    def linreg(y):
        if len(y)<L: return np.nan
        mean_y = np.mean(y)
        beta = np.mean((x - mean_x) * (y - mean_y)) / var_x
        return mean_y - beta * mean_x + beta * (L - 1)
    linreg_curve = norm_src.rolling(window=L).apply(linreg, raw=True)
    hw_osc = linreg_curve.ewm(span=3, adjust=False).mean()
    hw_sig = hw_osc.rolling(window=3).mean()
    df['Lux_Osc'] = hw_osc; df['Lux_Sig'] = hw_sig
    
    # MFI
    typical_price = hl2
    raw_money_flow = typical_price * vol
    pos_flow = np.where(typical_price > typical_price.shift(1), raw_money_flow, 0)
    neg_flow = np.where(typical_price < typical_price.shift(1), raw_money_flow, 0)
    pos_sum = pd.Series(pos_flow, index=df.index).rolling(window=35).sum()
    neg_sum = pd.Series(neg_flow, index=df.index).rolling(window=35).sum()
    mfi = 100 - (100 / (1 + pos_sum / neg_sum)) - 50
    df['Lux_MFI'] = mfi.rolling(window=6).mean()
    
    # ATR & CHOP & ADX
    tr_val = np.maximum(high - low, np.maximum((high - close.shift(1)).abs(), (low - close.shift(1)).abs()))
    df['ATR'] = tr_val.rolling(window=14).mean()
    
    tr1 = pd.DataFrame(index=df.index)
    tr1['h_l'] = high - low; tr1['h_pc'] = abs(high - close.shift(1)); tr1['l_pc'] = abs(low - close.shift(1))
    sum_tr = tr1.max(axis=1).rolling(window=14).sum()
    RANGE = (high.rolling(window=14).max() - low.rolling(window=14).min()).replace(0, np.nan)
    df['CHOP'] = 100 * np.log10(sum_tr / RANGE) / np.log10(14)
    
    up = high - high.shift(1); down = low.shift(1) - low
    plus = np.where((up > down) & (up > 0), up, 0.0)
    minus = np.where((down > up) & (down > 0), down, 0.0)
    smooth_tr = pd.Series(tr_val, index=df.index).ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * pd.Series(plus, index=df.index).ewm(alpha=1/14, adjust=False).mean() / smooth_tr
    minus_di = 100 * pd.Series(minus, index=df.index).ewm(alpha=1/14, adjust=False).mean() / smooth_tr
    df['ADX'] = (100 * abs(plus_di - minus_di) / (plus_di + minus_di)).ewm(alpha=1/14, adjust=False).mean()
    
    # --- Moto 1 Features ---
    sma50 = close.rolling(50).mean()
    df['Dist_SMA50'] = (close - sma50) / sma50
    
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # ML Features
    df['Osc_Slope'] = df['Lux_Osc'].diff()
    df['Sig_Slope'] = df['Lux_Sig'].diff()
    df['Ret'] = df['close'].pct_change()
    df['Vol_20'] = df['Ret'].rolling(20).std()
    df['Vol_SMA_20'] = df['volume'].rolling(20).mean()
    df['Rel_Vol'] = df['volume'] / (df['Vol_SMA_20'] + 1e-6)
    
    return df

# --- Trade Class (V4.1 Logic + Live Execution + DB Logging) ---
from trade.executor import execute_trade_sequence, update_stop_loss

class LiveTrade:
    def __init__(self, entry_time, entry_price, size, direction, atr, prob=0.0, reason="UNKNOWN", features=None):
        self.entry_time = str(entry_time)
        self.entry_price = float(entry_price)
        self.initial_size = float(size)
        self.current_size = float(size)
        self.direction = int(direction)
        self.active = True
        self.pnl_realized = 0.0
        
        # V4.1 Params (Tight)
        SL_ATR_MULT = 1.5
        TP_RR_RATIOS = [1.5, 3.0]
        TP_SIZES = [0.5, 0.5]
        
        self.sl_dist = float(atr * SL_ATR_MULT)
        self.prob = prob
        self.reason = reason
        self.features = features
        
        self.type_str = "BUY" if direction == 1 else "SELL"
        
        if self.direction == 1:
            self.sl_price = round(self.entry_price - self.sl_dist, 2)
            self.tps = [round(self.entry_price + (self.sl_dist * rr), 2) for rr in TP_RR_RATIOS]
        else:
            self.sl_price = round(self.entry_price + self.sl_dist, 2)
            self.tps = [round(self.entry_price - (self.sl_dist * rr), 2) for rr in TP_RR_RATIOS]
            
        self.tp_hit_count = 0
        
        # Logging
        print(f"\n[TRADE ENTRY] {self.type_str} @ {self.entry_price:.2f}")
        print(f"   Size: {self.initial_size:.4f} | SL: {self.sl_price:.2f}")
        print(f"   TP Goals: {[f'{x:.2f}' for x in self.tps]}")
        print(f"   Reason: {reason} | Confidence: {prob:.2f}")
        if features:
            print(f"   Features: {features}")
    
        # --- EXECUTE TRADE ---
        print("🚀 Executing trade via Executor...")
        
        # Run async executor in sync context
        try:
            order_ids = asyncio.run(execute_trade_sequence(
                market_id=MARKET_ID,
                direction=self.direction,
                size=self.initial_size,
                entry_price=self.entry_price,
                sl_price=self.sl_price,
                tps_list=self.tps,
                tp_sizes_list=TP_SIZES,
                leverage=LEVERAGE
            ))
            
            if order_ids:
                print("✅ Trade placed successfully on Lighter!")
                real_entry_id = str(order_ids.get('entry', 'ERROR'))
                real_sl_id = str(order_ids.get('stop_loss', 'ERROR'))
                self.sl_order_id = int(real_sl_id) if real_sl_id.isdigit() else None
            else:
                print("❌ Trade execution failed (returned False).")
                real_entry_id = "FAILED"
                real_sl_id = "FAILED"
                self.sl_order_id = None
                
        except Exception as e:
            print(f"❌ Executor Error: {e}")
            real_entry_id = "ERROR"
            real_sl_id = "ERROR"
            self.sl_order_id = None
        
        # 1. Log to 'placed_trades' (Execution Details)
        try:
            insert_placed_trade(
                symbol=SYMBOL,
                direction=self.type_str,
                entry_price=self.entry_price,
                size=self.initial_size,
                leverage=LEVERAGE,
                entry_order_id=real_entry_id,
                sl_price=self.sl_price,
                sl_order_id=real_sl_id,
                tps_details=self.tps,
                status="ACTIVE" if real_entry_id not in ["FAILED", "ERROR"] else "FAILED"
            )
        except Exception as e:
            print(f"DB Error (Placed): {e}")

        # 2. Log to 'trade_logs' (Signal/Strategy Details)
        try:
            now = datetime.datetime.now()
            insert_trade_log(
                predicted_trade=self.type_str,
                trade_date=str(now.date()),
                trade_time=str(now.time()),
                sl=self.sl_price,
                tps=self.tps
            )
        except Exception as e:
             print(f"DB Error (Log): {e}")

    def update(self, timestamp, high, low, close):
        if not self.active: return
        
        global current_balance
        
        # SL Check
        sl_hit = False
        exit_price = 0
        if self.direction == 1:
            if low <= self.sl_price: sl_hit = True; exit_price = self.sl_price
        else:
            if high >= self.sl_price: sl_hit = True; exit_price = self.sl_price
        if sl_hit: 
            self.close(timestamp, exit_price, "STOP LOSS HIT")
            return

        # TP Check
        TP_SIZES = [0.5, 0.5]
        while self.tp_hit_count < len(self.tps):
            tp_idx = self.tp_hit_count
            tp_price = self.tps[tp_idx]; tp_hit = False
            if self.direction == 1:
                if high >= tp_price: tp_hit = True
            else:
                if low <= tp_price: tp_hit = True
            if tp_hit: 
                amount = min(self.initial_size * TP_SIZES[tp_idx], self.current_size)
                pnl = (tp_price - self.entry_price) * amount if self.direction == 1 else (self.entry_price - tp_price) * amount
                self.pnl_realized += pnl
                self.current_size -= amount
                self.tp_hit_count += 1
                
                # Update global balance directly
                current_balance += pnl
                
                # Move SL
                if tp_idx == 0: self.sl_price = self.entry_price # BE
                elif tp_idx == 1: self.sl_price = (self.entry_price + self.tps[0])/2
                
                # Update SL on Exchange
                try:
                    print(f"🔄 Updating SL on Lighter to {self.sl_price}...")
                    new_id = asyncio.run(update_stop_loss(
                        market_id=MARKET_ID,
                        direction=self.direction,
                        new_sl_price=self.sl_price,
                        size=self.current_size,
                        leverage=LEVERAGE,
                        old_sl_order_index=self.sl_order_id
                    ))
                    if new_id:
                        self.sl_order_id = new_id
                        print(f"✅ SL Updated on Exchange: {new_id}")
                except Exception as e:
                    print(f"❌ Failed to update SL on exchange: {e}") 
                
                # Logging
                print(f"[{timestamp}] ✅ TP-{tp_idx+1} HIT @ {tp_price:.2f} | PnL: ${pnl:.2f}")
                print(f"   New Balance: ${current_balance:.2f} | SL Moved to: {self.sl_price:.2f}")
                
                if self.current_size <= 1e-6: 
                    self.active = False
                    print(f"[{timestamp}] 🏆 Trade Closed (All TPs Hit). Total PnL: ${self.pnl_realized:.2f}")
            else: break
            
    def close(self, timestamp, price, reason):
        global current_balance
        if not self.active: return
        
        pnl = (price - self.entry_price) * self.current_size if self.direction == 1 else (self.entry_price - price) * self.current_size
        self.pnl_realized += pnl
        current_balance += pnl
        self.current_size = 0
        self.active = False
        
        # Logging
        print(f"[{timestamp}] ❌ {reason} @ {price:.2f} | PnL: ${pnl:.2f}")
        print(f"   Final Balance: ${current_balance:.2f}")

# --- Helper Functions ---
def fetch_candles_rest(count_back=500):
    """ Fetch historical candles for warmup """
    print(f"Fetching {count_back} historical candles...")
    
    end_ts = int(time.time() * 1000)
    
    url = f"{LIGHTER_REST_URL}/candles"
    params = {
        "market_id": MARKET_ID,
        "resolution": TIMEFRAME,
        "end_timestamp": end_ts,
        "count_back": count_back,
        "start_timestamp": 0
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if 'c' in data and len(data['c']) > 0:
            batch = data['c']
            candles = []
            for c in batch:
                candles.append({
                    "timestamp": pd.to_datetime(c['t'], unit='ms'),
                    "open": float(c['o']),
                    "high": float(c['h']),
                    "low": float(c['l']),
                    "close": float(c['c']),
                    "volume": float(c['v'])
                })
            df = pd.DataFrame(candles)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            return df
    except Exception as e:
        print(f"Error fetching candles: {e}")
    
    return pd.DataFrame()

def build_candle_from_trades(trades, candle_start_time):
    if not trades: return None
    try:
        prices = [float(t['price']) for t in trades]
        volumes = [float(t['size']) for t in trades]
        return {
            'open': prices[0], 'high': max(prices), 'low': min(prices),
            'close': prices[-1], 'volume': sum(volumes)
        }
    except: return None

# --- Main Logic ---
def analyze_candle(i, df):
    global active_trade
    
    if i < 100: return  # V4.1 needs 100 candles minimum
    
    curr = df.iloc[i]; timestamp = curr.name
    close_vals = df['close'].values
    high_vals = df['high'].values
    low_vals = df['low'].values
    atr_vals = df['ATR'].values
    
    # Feature Arrays
    trend_vals = df['SmartTrail_Trend'].values
    mfi_vals = df['Lux_MFI'].values
    adx_vals = df['ADX'].values
    sma_dist_vals = df['Dist_SMA50'].values
    rsi_vals = df['RSI'].values
    
    # 1. Update Active Trade
    if active_trade and active_trade.active:
        active_trade.update(timestamp, high_vals[i], low_vals[i], close_vals[i])
        if not active_trade.active:
            active_trade = None
        return
    
    # 2. Logic Check (Moto 2 - Relaxed Entry if ADX strong)
    is_candidate_long = False
    is_candidate_short = False
    
    # --- LONG ---
    # Base: Trend UP and MFI OK
    # Relaxed: OR ADX > 30 and MFI > -25
    if trend_vals[i] == 1:
        if mfi_vals[i] > -10: is_candidate_long = True
        elif adx_vals[i] > ADX_STRONG and mfi_vals[i] > MFI_RELAXED_LONG: is_candidate_long = True
        
    # --- SHORT ---
    if trend_vals[i] == -1:
        if mfi_vals[i] < 10: is_candidate_short = True
        elif adx_vals[i] > ADX_STRONG and mfi_vals[i] < MFI_RELAXED_SHORT: is_candidate_short = True
        
    # 3. Moto 1 - Loss Shield (Hard Filters) + Shield V2
    if is_candidate_long:
        if sma_dist_vals[i] > MAX_SMA_DIST: is_candidate_long = False 
        if rsi_vals[i] > MAX_RSI_LONG: is_candidate_long = False 
        # Shield V2: ADX Gap
        if 25 < adx_vals[i] < 30: is_candidate_long = False
        
    if is_candidate_short:
        if sma_dist_vals[i] < -MAX_SMA_DIST: is_candidate_short = False 
        if rsi_vals[i] < MIN_RSI_SHORT: is_candidate_short = False
        # Shield V2: ADX Gap & MFI Oversold
        if 25 < adx_vals[i] < 30: is_candidate_short = False
        if mfi_vals[i] < -10: is_candidate_short = False # Don't short if MFI < -10 (Oversold/Extension)
        
    # 4. ML Validation (Standard V4.1)
    direction = 0
    prob = 0.0
    
    if (is_candidate_long or is_candidate_short) and model:
        # Construct Features
        feats = {
            "MFI": mfi_vals[i], "Osc": curr['Lux_Osc'], "Sig": curr['Lux_Sig'],
            "Osc_Slope": curr['Osc_Slope'], "Sig_Slope": curr['Sig_Slope'],
            "Trend": trend_vals[i], "ADX": curr['ADX'], "CHOP": curr['CHOP'],
            "Vol_20": curr['Vol_20'], "Rel_Vol": curr['Rel_Vol'],
            "Direction": 1 if is_candidate_long else -1, "Hour": timestamp.hour
        }
        # Add Lags
        for k in range(1, 6): feats[f"Lag_{k}"] = (close_vals[i-k] - close_vals[i]) / close_vals[i] * 100
        
        try:
            X_in = pd.DataFrame([feats])
            X_scaled = scaler.transform(X_in)
            p = model.predict_proba(X_scaled)[0][1]
            
            print(f"[{timestamp}] Checking V4.1 ML for {('BUY' if is_candidate_long else 'SELL')}...")
            print(f"   ML Confidence: {p:.2f}")
            
            if p >= CONFIDENCE_THRESHOLD:
                if is_candidate_long: 
                    direction = 1; prob = p
                else: 
                    direction = -1; prob = p
                print(f"!!! ML V4.1 APPROVED !!!")
            else:
                print("   Rejected by ML.")
        except Exception as e:
            print(f"   ML Check Error: {e}")
        
    if direction != 0:
        size = (INITIAL_CAPITAL * LEVERAGE) / close_vals[i]
        reason_str = "BASE"
        if is_candidate_long:
             if mfi_vals[i] <= -10: reason_str = "SWORD_ADX"
        else: # Short
             if mfi_vals[i] >= 10: reason_str = "SWORD_ADX"
        
        feat_log = {
            "MFI": float(mfi_vals[i]), "ADX": float(adx_vals[i]), 
            "SMA_Dist": float(sma_dist_vals[i]), "RSI": float(rsi_vals[i])
        }
        active_trade = LiveTrade(timestamp, close_vals[i], size, direction, atr_vals[i], prob, reason=reason_str, features=feat_log)

def main():
    global candles_df, active_trade
    
    load_ml_model()
    
    try:
        init_db()
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Database initialized")
    except Exception as e:
        print(f"DB Init Failed: {e}")

    print(f"Live Bot V4.1 (Moto Shield V2) Started. \nBalance: ${current_balance:.2f} | Leverage: {LEVERAGE}x")
    
    # Warmup
    candles_df = fetch_candles_rest(count_back=500)
    if candles_df.empty:
        print("Failed to fetch initial data. Exiting.")
        return
        
    # Calculate initial indicators
    print("Calculating initial indicators...")
    candles_df = full_indicator_calc(candles_df).dropna()
    print(f"Ready with {len(candles_df)} candles.")
    
    # Run analysis on latest candle (optional, in case we just started at candle close)
    analyze_candle(len(candles_df)-1, candles_df)
    
    # WebSocket Logic
    last_ws_update_time = time.time()
    last_candle_time = None
    trade_buffer = []
    ws = None
    ws_connected = False
    
    print(f"\n--- LIVE MONITORING V4.1 (WebSocket) ---")
    print(f"Current Balance: ${current_balance:.2f}")
    if active_trade: print(f"Active Trade: {active_trade.type_str} (PnL: {active_trade.pnl_realized:.2f})")

    # WebSocket Handlers (Nested to access scope)
    def on_trade_update(msg):
        nonlocal last_candle_time, trade_buffer, last_ws_update_time
        global candles_df
        last_ws_update_time = time.time()
        
        try:
            channel = msg.get('channel', '')
            if not channel.startswith('trade:'): return
            
            trades_data = msg.get('trades', [])
            if not trades_data: return
            
            for trade_data in trades_data:
                price = float(trade_data.get('price', 0))
                size = float(trade_data.get('size', 0))
                ts_ms = trade_data.get('timestamp', int(time.time() * 1000))
                
                if price == 0 or size == 0: continue
                
                trade_ts = pd.to_datetime(ts_ms, unit='ms')
                candle_time = trade_ts.floor(f'{TIMEFRAME_MINUTES}min')
                
                if last_candle_time is None:
                    last_candle_time = candle_time
                    trade_buffer = []
                    
                if candle_time > last_candle_time:
                    if trade_buffer:
                        completed = build_candle_from_trades(trade_buffer, last_candle_time)
                        if completed:
                            row = pd.DataFrame([completed], index=[last_candle_time])
                            candles_df = pd.concat([candles_df, row])
                            
                            # Keep buffer manageable
                            if len(candles_df) > 1000: candles_df = candles_df.iloc[-1000:]
                            
                            print(f"\n{'='*80}")
                            print(f"CANDLE CLOSED: {last_candle_time}")
                            print(f"  O: {completed['open']:.2f} | H: {completed['high']:.2f} | L: {completed['low']:.2f} | C: {completed['close']:.2f} | V: {completed['volume']:.4f}")
                            print(f"{'='*80}\n")
                            
                            if len(candles_df) >= 100: # V4.1 needs 100 candles minimum
                                # Recalc and Analyze
                                candles_df = full_indicator_calc(candles_df)
                                analyze_candle(len(candles_df)-1, candles_df)
                            else:
                                print(f"Building history... {len(candles_df)}/100 candles")
                            
                    last_candle_time = candle_time
                    trade_buffer = []
                
                trade_buffer.append({'price': price, 'size': size, 'timestamp': trade_ts})
                
                # Live PnL Update 
                if active_trade and active_trade.active:
                     pass
                     
                if trade_buffer:
                     forming_candle = build_candle_from_trades(trade_buffer, candle_time)
                     if forming_candle:
                          sys.stdout.write(f"\r[{datetime.datetime.now().strftime('%H:%M:%S')}] Forming Candle: O:{forming_candle['open']:.2f} H:{forming_candle['high']:.2f} L:{forming_candle['low']:.2f} C:{forming_candle['close']:.2f} V:{forming_candle['volume']:.4f} | Trades: {len(trade_buffer)}     ")
                          sys.stdout.flush()
                     
        except Exception as e:
            print(f"\nError processing trade update: {e}")

    def on_message(ws, message):
         try:
             msg = json.loads(message)
             if 'type' in msg:
                 msg_type = msg.get('type', '')
                 if msg_type == 'subscribed':
                    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Subscription confirmed: {msg.get('channel', '')}")
                    return
                 elif msg_type == 'error':
                    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Error from server: {msg}")
                    return
                 elif msg_type == 'update/trade':
                     on_trade_update(msg)

         except json.JSONDecodeError as e:
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] JSON decode error: {e}")
         except Exception as e:
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Error in on_message: {e}")
            
    def on_error(ws, error):
        print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] WebSocket Error: {error}")
    
    def on_close(ws, close_status_code, close_msg):
        nonlocal ws_connected
        ws_connected = False
        print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] WebSocket Closed: {close_status_code} - {close_msg}")
             
    def on_open(ws):
        nonlocal ws_connected
        ws_connected = True
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] WebSocket Connected to Lighter")
        sub = {"type": "subscribe", "channel": f"trade/{MARKET_ID}"}
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Subscribing to: trade/{MARKET_ID}")
        ws.send(json.dumps(sub))
        
    def connect_ws():
        nonlocal ws
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Connecting to Lighter WebSocket...")
        
        ws = websocket.WebSocketApp(
            LIGHTER_WS_URL,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        t = threading.Thread(target=ws.run_forever)
        t.daemon = True
        t.start()
        return ws
        
    ws = connect_ws()
    last_heartbeat = time.time()
    
    while not stop_event.is_set():
        try:
            time.sleep(1)
            
            # Reconnection logic
            if not ws_connected or (time.time() - last_ws_update_time > 30):
                print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] WebSocket Disconnected (No updates for 30s). Reconnecting...")
                if ws: ws.close()
                ws = connect_ws()
                last_ws_update_time = time.time()
            
            # Heartbeat logic
            if time.time() - last_heartbeat > 60:
                active_trade_status = 'YES' if active_trade and active_trade.active else 'NO'
                candles_count = len(candles_df)
                print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] STATUS OK | Candles: {candles_count} | Balance: ${current_balance:.2f} | Active Trade: {active_trade_status}")
                last_heartbeat = time.time()

        except KeyboardInterrupt:
            print("\n\nStopped by user.")
            if ws: ws.close()
            break
        except Exception as e:
            print(f"\nError in main loop: {e}")
            import traceback
            traceback.print_exc()
            traceback.print_exc()
            time.sleep(5)
            
    if ws: ws.close()
    print("Bot loop finished.")

# --- Signal handlers for graceful shutdown ---
stop_event = threading.Event()

def signal_handler(signum, frame):
    print("\n⚠️  Received shutdown signal. Gracefully stopping...")
    stop_event.set()

if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🚀 Starting Scalping Trading Agent...")
    print(f"Environment: K8s Pod")
    print(f"User API Key Index: {os.getenv('LIGHTER_API_KEY_INDEX', 'Not Set')}")
    print(f"User Account Index: {os.getenv('LIGHTER_ACCOUNT_INDEX', 'Not Set')}")
    print(f"Database URL: {'Set' if os.getenv('DATABASE_URL') else 'Not Set'}")
    
    try:
        main()
    except Exception as e:
        print(f"❌ Bot execution failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("🛑 Trading Agent stopped.")
