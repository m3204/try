import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
import yfinance as yf

# Fetch data
def fetch_data(ticker='SPY', period='1y', interval='15m'):
    return yf.download(ticker, period=period, interval=interval)['Close']

# Calculate Hurst Exponent (simplified)
def hurst_exponent(prices, max_lag=20):
    lags = range(2, max_lag)
    rs = []
    for lag in lags:
        diff = prices.diff(lag).dropna()
        r = diff.max() - diff.min()
        s = diff.std()
        rs.append(np.log(r / s) / np.log(lag))
    return np.mean(rs)

# Calculate indicators
def calculate_indicators(data, lookback=20):
    data['Mean'] = data.rolling(window=lookback).mean()
    data['Std'] = data.rolling(window=lookback).std()
    data['Kurtosis'] = data.rolling(window=lookback).apply(kurtosis, raw=True)
    data['Skew'] = data.rolling(window=lookback).apply(skew, raw=True)
    data['Z_adj'] = (data - data['Mean']) / (data['Std'] * np.sqrt(1 + data['Kurtosis'] / 3))
    data['Hurst'] = data.rolling(window=lookback).apply(hurst_exponent, raw=True)
    return data

# Trading strategy
def trading_strategy(data, account_balance):
    data = calculate_indicators(data)
    position = None
    trades = []
    entry_price = 0
    entry_bar = 0
    
    for i in range(lookback, len(data)):
        z_adj = data['Z_adj'].iloc[i]
        hurst = data['Hurst'].iloc[i]
        skew_val = data['Skew'].iloc[i]
        price = data.iloc[i]
        
        if position is None:
            if z_adj < -2 and hurst < 0.5 and (skew_val > 0.5 or abs(skew_val) < 0.5):
                position = 'long'
                entry_price = price
                entry_bar = i
                print(f"Buy at {entry_price}")
            elif z_adj > 2 and hurst < 0.5 and (skew_val < -0.5 or abs(skew_val) < 0.5):
                position = 'short'
                entry_price = price
                entry_bar = i
                print(f"Sell at {entry_price}")
        
        elif position == 'long':
            if abs(z_adj) < 0.5 or z_adj > 3.5 or (i - entry_bar >= 10):
                profit = price - entry_price
                trades.append(profit)
                print(f"Exit long at {price}, Profit: {profit}")
                position = None
        elif position == 'short':
            if abs(z_adj) < 0.5 or z_adj < -3.5 or (i - entry_bar >= 10):
                profit = entry_price - price
                trades.append(profit)
                print(f"Exit short at {price}, Profit: {profit}")
                position = None
    
    return trades

# Monte Carlo Refinement (from previous)
def monte_carlo_refinement(historical_prices, param_sets, num_paths=1000):
    # Same as previous, just update simulate_trades to use new indicators
    pass  # Reuse earlier code with adjusted logic

# Execute
if __name__ == "__main__":
    data = fetch_data('SPY')
    trades = trading_strategy(data, 10000)
    print(f"Total Trades: {len(trades)}, Total Profit: {sum(trades):.2f}")
