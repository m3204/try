import pandas as pd
import numpy as np
from tvDatafeed import TvDatafeed, Interval
import talib as ta
from matplotlib import pyplot as plt

# Initialize TvDatafeed
tv = TvDatafeed()

# Load SPY and VIX data
spy = tv.get_hist(symbol="SPY", exchange="AMEX", interval=Interval.in_5_minute, n_bars=100000)
vix = tv.get_hist(symbol="VIX", exchange="CBOE", interval=Interval.in_5_minute, n_bars=100000)

# Rename columns for consistency
spy.rename(columns={'close': 'Close', 'open': 'Open', 'high': 'High', 'low': 'Low', 'volume': 'Volume'}, inplace=True)
vix.rename(columns={'close': 'VIX_Close'}, inplace=True)

# Merge SPY and VIX data
df = spy[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
df['VIX'] = vix['VIX_Close']

# Calculate returns and features
df['Returns'] = df['Close'].pct_change()

# Trend Strength (ADX)
df['ADX'] = ta.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)

# Momentum (RSI and MACD)
df['RSI'] = ta.RSI(df['Close'], timeperiod=14)
df['MACD'], df['MACD_Signal'], _ = ta.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

# Volatility (ATR and Bollinger Bands)
df['ATR'] = ta.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
df['Upper_Band'], df['Middle_Band'], df['Lower_Band'] = ta.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)

# Volume Confirmation
df['Volume_MA'] = df['Volume'].rolling(20).mean()
df['Volume_Surge'] = (df['Volume'] > 2 * df['Volume_MA']).astype(int)

# Bayesian Probability Calculation
def update_bayesian_probability(df):
    window = 252  # 1-hour bars (~2 months)
    
    # Lag features to prevent look-ahead
    df['RSI_Lagged'] = df['RSI'].shift(1)
    df['ADX_Lagged'] = df['ADX'].shift(1)
    df['MACD_Lagged'] = df['MACD'].shift(1)
    df['Volume_Surge_Lagged'] = df['Volume_Surge'].shift(1)
    
    # Dynamic rolling likelihoods
    df['Prior_Up'] = df['Close'].gt(df['Close'].shift(1)).rolling(window).mean().shift(1)
    
    # Up-day likelihood components
    df['rsi_up'] = (df['RSI_Lagged'] > 50).rolling(window).mean().shift(1)
    df['adx_up'] = (df['ADX_Lagged'] > 25).rolling(window).mean().shift(1)
    df['macd_up'] = (df['MACD_Lagged'] > df['MACD_Signal'].shift(1)).rolling(window).mean().shift(1)
    # df['vol_up'] = df['Volume_Surge_Lagged'].rolling(window).mean().shift(1)
    
    # Down-day likelihood components
    df['rsi_down'] = (df['RSI_Lagged'] < 50).rolling(window).mean().shift(1)
    df['adx_down'] = (df['ADX_Lagged'] < 20).rolling(window).mean().shift(1)
    df['macd_down'] = (df['MACD_Lagged'] < df['MACD_Signal'].shift(1)).rolling(window).mean().shift(1)
    # df['vol_down'] = (1 - df['Volume_Surge_Lagged']).rolling(window).mean().shift(1)
    
    # Posterior calculation
    # numerator_up = df['rsi_up'] * df['adx_up'] * df['macd_up'] * df['vol_up'] * df['Prior_Up']
    # p_features = numerator_up + (df['rsi_down'] * df['adx_down'] * df['macd_down'] * df['vol_down'] * (1 - df['Prior_Up']))
    # df['Posterior_Up'] = numerator_up / p_features

    # numerator_up = df['rsi_up'] * df['adx_up'] * df['macd_up'] * df['Prior_Up']
    # p_features = numerator_up + (df['rsi_down'] * df['adx_down'] * df['macd_down'] * (1 - df['Prior_Up']))
    # df['Posterior_Up'] = numerator_up / p_features

    numerator_up = df['rsi_up'] * df['macd_up'] * df['Prior_Up']
    p_features = numerator_up + (df['rsi_down'] * df['macd_down'] * (1 - df['Prior_Up']))
    df['Posterior_Up'] = numerator_up / p_features

    return df

# Update Bayesian probabilities
df = update_bayesian_probability(df)

df['Posterior_Up'].plot(kind='hist')

# Generate signals
df['Signal'] = 0
df.loc[(df['Posterior_Up'] > 0.6) & (df['ADX'] > 30), 'Signal'] = 1  # Long
df.loc[(df['Posterior_Up'] < 0.4) & (df['ADX'] < 20), 'Signal'] = -1  # Short

# df.loc[(df['Posterior_Up'] > 0.6), 'Signal'] = 1  # Long
# df.loc[(df['Posterior_Up'] < 0.4), 'Signal'] = -1  # Short


# Backtest with risk rules
capital = 100_000
position = 0
equity_curve = []

ts = {
    'EntryType': {},
    'EntryPrice': {},
    'ExitPrice': {},
    'ExitTime': {},
    'PL': {}
}

for i in range(len(df)):
    atr = df['ATR'].iloc[i]
    close = df['Close'].iloc[i]
    
    # Entry
    if df['Signal'].iloc[i] == 1 and position == 0:
        position_size = (0.01 * capital) / (2.5 * atr)  # Risk 1% per trade
        entry_price = close
        stop_loss = entry_price - 2 * atr  # 2x ATR stop
        take_profit = entry_price + 4 * atr  # 4x ATR target
        position = 1
        entry_index = df.index[i]
        ts['EntryType'][entry_index] = 'LONG'
        ts['EntryPrice'][entry_index] = entry_price
    
    elif df['Signal'].iloc[i] == -1 and position == 0:
        position_size = (0.01 * capital) / (2.5 * atr)  # Risk 1% per trade
        entry_price = close
        stop_loss = entry_price + 1.5 * atr  # 1.5x ATR stop
        take_profit = entry_price - 3 * atr  # 3x ATR target
        position = -1
        entry_index = df.index[i]
        ts['EntryType'][entry_index] = 'SHORT'
        ts['EntryPrice'][entry_index] = entry_price
    
    # Exit
    if position == 1:
        if df['Low'].iloc[i] <= stop_loss or df['High'].iloc[i] >= take_profit:
            exit_price = take_profit if df['High'].iloc[i] >= take_profit else stop_loss
            pnl = (exit_price - entry_price) / entry_price * position_size
            capital += pnl
            position = 0
            ts['ExitPrice'][entry_index] = exit_price
            ts['ExitTime'][entry_index] = df.index[i]
            ts['PL'][entry_index] = ts['ExitPrice'][entry_index] - ts['EntryPrice'][entry_index]
    
    elif position == -1:
        if df['High'].iloc[i] >= stop_loss or df['Low'].iloc[i] <= take_profit:
            exit_price = take_profit if df['Low'].iloc[i] <= take_profit else stop_loss
            pnl = (entry_price - exit_price) / entry_price * position_size
            capital += pnl
            position = 0
            ts['ExitPrice'][entry_index] = exit_price
            ts['ExitTime'][entry_index] = df.index[i]
            ts['PL'][entry_index] = ts['EntryPrice'][entry_index] - ts['ExitPrice'][entry_index]
    
    equity_curve.append(capital)

# Convert trades to DataFrame
ts = pd.DataFrame(ts)

# Performance Metrics
long_trades = ts[ts['EntryType'] == 'LONG']
short_trades = ts[ts['EntryType'] == 'SHORT']

print("Long Trades:")
print(f"Total: {len(long_trades)}")
print(f"Win Rate: {len(long_trades[long_trades['PL'] > 0]) / len(long_trades):.2%}")
print(f"Avg Profit: {long_trades['PL'].mean():.2f}")
print(f"Total PnL: {long_trades['PL'].sum():.2f}")

print("\nShort Trades:")
print(f"Total: {len(short_trades)}")
print(f"Win Rate: {len(short_trades[short_trades['PL'] > 0]) / len(short_trades):.2%}")
print(f"Avg Profit: {short_trades['PL'].mean():.2f}")
print(f"Total PnL: {short_trades['PL'].sum():.2f}")
