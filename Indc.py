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
