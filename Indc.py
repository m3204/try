import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the stock symbol and time period
symbol = 'SPY'  
start_date = '2020-01-01' 
end_date = '2025-01-01'

# Download the data
df = yf.download(symbol, start=start_date, end=end_date)
df.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
df.ffill(inplace=True)

# ---- Gopalakrishnan Range Index (GAPO) ----
def gapo(df, period=14):
    df['range'] = df['High'] - df['Low']
    df['max_range'] = df['range'].rolling(window=period).max()
    df['GAPO'] = df['range'] / df['max_range'] * 100
    return df

# ---- True Strength Index (TSI) ----
def tsi(df, short_period=25, long_period=13):
    diff = df['Close'].diff()
    abs_diff = diff.abs()
    momentum = diff.ewm(span=short_period).mean() / abs_diff.ewm(span=long_period).mean()
    df['TSI'] = momentum * 100
    return df

# ---- Know Sure Thing (KST) ----
def kst(df, long_period=34, short_period=23, signal_period=10):
    roc1 = df['Close'].pct_change(periods=10) * 100
    roc2 = df['Close'].pct_change(periods=15) * 100
    roc3 = df['Close'].pct_change(periods=20) * 100
    roc4 = df['Close'].pct_change(periods=30) * 100
    
    kst = (roc1.rolling(window=long_period).mean() +
           roc2.rolling(window=short_period).mean() +
           roc3.rolling(window=short_period).mean() +
           roc4.rolling(window=long_period).mean())
    
    df['KST'] = kst
    df['KST_signal'] = kst.rolling(window=signal_period).mean()
    return df

# ---- Elder Force Index (EFI) ----
def efi(df, period=13):
    df['EFI'] = (df['Close'].diff(periods=1) * df['Volume']) / df['Close']
    df['EFI'] = df['EFI'].rolling(window=period).mean()
    return df

# ---- Trend Intensity Index (TII) ----
def tii(df, period=14):
    df['trend'] = df['Close'] - df['Close'].shift(period)
    df['trend_intensity'] = df['trend'] / df['Close'].rolling(window=period).std()
    return df

# Apply each function to the DataFrame
df = gapo(df)
df = tsi(df)
df = kst(df)
df = efi(df)
df = tii(df)

# Plot the indicators
fig, axs = plt.subplots(5, 1, figsize=(10, 15))

# GAPO Plot
axs[0].plot(df.index, df['GAPO'], label='GAPO', color='blue')
axs[0].set_title('Gopalakrishnan Range Index (GAPO)')
axs[0].legend()

# TSI Plot
axs[1].plot(df.index, df['TSI'], label='TSI', color='green')
axs[1].set_title('True Strength Index (TSI)')
axs[1].legend()

# KST Plot
axs[2].plot(df.index, df['KST'], label='KST', color='red')
axs[2].plot(df.index, df['KST_signal'], label='KST Signal', color='orange')
axs[2].set_title('Know Sure Thing (KST)')
axs[2].legend()

# EFI Plot
axs[3].plot(df.index, df['EFI'], label='EFI', color='purple')
axs[3].set_title('Elder Force Index (EFI)')
axs[3].legend()

# TII Plot
axs[4].plot(df.index, df['trend_intensity'], label='TII', color='brown')
axs[4].set_title('Trend Intensity Index (TII)')
axs[4].legend()

# Display the plot
plt.tight_layout()
plt.show()


# new code

# Function: Polynomial Regression Moving Average (PRMA)
def prma(series, window, degree=2):
    return [np.polyval(np.polyfit(range(window), series[i:i+window], degree), window-1)
            for i in range(len(series) - window + 1)]

# Function: Chande Momentum Oscillator (CMO)
def chande_momentum_oscillator(df, period=14):
    df['Up'] = np.where(df['Close'] > df['Close'].shift(1), df['Close'] - df['Close'].shift(1), 0)
    df['Down'] = np.where(df['Close'] < df['Close'].shift(1), df['Close'].shift(1) - df['Close'], 0)
    df['SumUp'] = df['Up'].rolling(window=period).sum()
    df['SumDown'] = df['Down'].rolling(window=period).sum()
    df['CMO'] = 100 * (df['SumUp'] - df['SumDown']) / (df['SumUp'] + df['SumDown'])
    return df
    
# Function to optimize parameters
def optimize_parameters(cmo_periods, fast_windows, slow_windows):
    best_total_return = -np.inf
    best_params = None
    best_portfolio = None

    # Grid search over CMO periods, fast PRMA windows, and slow PRMA windows
    param_grid = ParameterGrid({
        'cmo_period': cmo_periods,
        'fast_window': fast_windows,
        'slow_window': slow_windows
    })

    for params in param_grid:
        # Reload the original dataframe for each iteration
        df_copy = df.copy()

        # Calculate CMO with current period
        df_copy = chande_momentum_oscillator(df_copy, period=params['cmo_period'])

        # Calculate PRMA with different windows
        fast_prma = prma(df_copy['Close'], window=params['fast_window'], degree=2)
        slow_prma = prma(df_copy['Close'], window=params['slow_window'], degree=2)

        # Align PRMA values to match original dataframe length
        fast_prma = [np.nan] * (params['fast_window'] - 1) + fast_prma
        slow_prma = [np.nan] * (params['slow_window'] - 1) + slow_prma

        df_copy['Fast_PRMA'] = fast_prma
        df_copy['Slow_PRMA'] = slow_prma

        # Define Entry and Exit Signals based on PRMA crossovers
        df_copy['Entry'] = (
            (df_copy['CMO'] < -50) &  # CMO is below -50
            (df_copy['Fast_PRMA'] > df_copy['Slow_PRMA'])  # Current Fast PRMA crosses above Slow PRMA
        )

        df_copy['Exit'] = (
            (df_copy['CMO'] > 50) &  # CMO is above +50
            (df_copy['Fast_PRMA'] < df_copy['Slow_PRMA'])  # Current Fast PRMA crosses below Slow PRMA
        )

        # Convert signals to boolean arrays
        entries = df_copy['Entry'].to_numpy()
        exits = df_copy['Exit'].to_numpy()

        # Backtest using vectorbt
        portfolio = vbt.Portfolio.from_signals(
            close=df_copy['Close'],
            entries=entries,
            exits=exits,
            init_cash=100_000,
            fees=0.001
        )

        # Calculate Total Return for performance
        total_return = portfolio.stats()['Total Return [%]']

        # Keep track of the best parameters based on Total Return
        if total_return > best_total_return:
            best_total_return = total_return
            best_params = params
            best_portfolio = portfolio

    return best_params, best_portfolio
