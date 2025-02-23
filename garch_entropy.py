import numpy as np
import pandas as pd
from scipy.stats import entropy, norm
from statsmodels.tsa.stattools import adfuller
import arch  # For GARCH
import yfinance as yf

# Fetch data
def fetch_data(ticker='SPY', period='1y', interval='15m'):
    df = yf.download(ticker, period=period, interval=interval)
    return df[['Close', 'Volume']]

# Shannon Entropy
def calc_entropy(returns, bins=20):
    hist, _ = np.histogram(returns.dropna(), bins=bins, density=True)
    return entropy(hist)

# GARCH(1,1) Volatility
def calc_garch_vol(returns):
    model = arch.arch_model(returns, vol='Garch', p=1, q=1)
    res = model.fit(disp='off')
    return res.conditional_volatility[-1]

# Copula (simplified Gaussian)
def calc_copula(price_z, vol_z, rho=0.5):
    u = norm.cdf(price_z)
    v = norm.cdf(vol_z)
    return norm.cdf((norm.ppf(u) - rho * norm.ppf(v)) / np.sqrt(1 - rho**2))

# Indicators
def calculate_indicators(data, lookback=20):
    prices = data['Close']
    volumes = data['Volume']
    returns = prices.pct_change().dropna()
    
    data['Mean'] = prices.rolling(lookback).mean()
    data['Std'] = prices.rolling(lookback).std()
    data['Kurtosis'] = prices.rolling(lookback).kurt()
    data['Skew'] = prices.rolling(lookback).skew()
    data['Z_adj'] = (prices - data['Mean']) / (data['Std'] * np.sqrt(1 + data['Kurtosis'] / 3))
    data['Hurst'] = prices.rolling(lookback).apply(lambda x: np.mean([np.log((x.max() - x.min()) / x.std()) / np.log(lookback)]), raw=True)
    data['Entropy'] = returns.rolling(lookback).apply(calc_entropy, raw=True)
    data['GARCH_Vol'] = returns.rolling(lookback).apply(calc_garch_vol, raw=True)
    data['ADF_p'] = prices.rolling(lookback).apply(lambda x: adfuller(x)[1], raw=True)
    data['Vol_Z'] = (volumes - volumes.rolling(lookback).mean()) / volumes.rolling(lookback).std()
    data['Copula'] = data.apply(lambda row: calc_copula(row['Z_adj'], row['Vol_Z']), axis=1)
    return data

# Trading strategy
def trading_strategy(data, account_balance):
    data = calculate_indicators(data)
    position = None
    trades = []
    bayes_p = 0.6  # Initial P(revert)
    
    for i in range(lookback, len(data)):
        z_adj = data['Z_adj'].iloc[i]
        hurst = data['Hurst'].iloc[i]
        skew = data['Skew'].iloc[i]
        entropy_val = data['Entropy'].iloc[i]
        garch_vol = data['GARCH_Vol'].iloc[i]
        copula = data['Copula'].iloc[i]
        adf_p = data['ADF_p'].iloc[i]
        price = data['Close'].iloc[i]
        
        entry_threshold = 2 + (garch_vol / data['Std'].iloc[i])  # Dynamic
        stop_threshold = 3.5 + (garch_vol / data['Std'].iloc[i])
        
        if position is None:
            if (z_adj < -entry_threshold and hurst < 0.5 and (skew > 0.5 or abs(skew) < 0.5) and 
                entropy_val < 2.5 and copula > 0.1 and adf_p < 0.05):
                position = 'long'
                entry_price = price
                entry_bar = i
                print(f"Buy at {entry_price}")
            elif (z_adj > entry_threshold and hurst < 0.5 and (skew < -0.5 or abs(skew) < 0.5) and 
                  entropy_val < 2.5 and copula > 0.1 and adf_p < 0.05):
                position = 'short'
                entry_price = price
                entry_bar = i
                print(f"Sell at {entry_price}")
        
        elif position == 'long':
            if abs(z_adj) < 0.5 or z_adj > stop_threshold or (i - entry_bar >= 10):
                profit = price - entry_price
                trades.append(profit)
                bayes_p = (0.8 * bayes_p) / (0.8 * bayes_p + 0.2 * (1 - bayes_p)) if profit > 0 else (0.2 * bayes_p) / (0.2 * bayes_p + 0.8 * (1 - bayes_p))
                print(f"Exit long at {price}, Profit: {profit}, P(revert): {bayes_p:.2f}")
                position = None
        elif position == 'short':
            if abs(z_adj) < 0.5 or z_adj < -stop_threshold or (i - entry_bar >= 10):
                profit = entry_price - price
                trades.append(profit)
                bayes_p = (0.8 * bayes_p) / (0.8 * bayes_p + 0.2 * (1 - bayes_p)) if profit > 0 else (0.2 * bayes_p) / (0.2 * bayes_p + 0.8 * (1 - bayes_p))
                print(f"Exit short at {price}, Profit: {profit}, P(revert): {bayes_p:.2f}")
                position = None
    
    return trades

# Execute
if __name__ == "__main__":
    data = fetch_data('SPY')
    trades = trading_strategy(data, 10000)
    print(f"Total Trades: {len(trades)}, Total Profit: {sum(trades):.2f}")
