import pandas as pd
import numpy as np
from statsmodels.tsa.filters.hp_filter import hpfilter

# Load OHLC data for Gold Futures
data = pd.read_csv("Gold_Futures_OHLC.csv")
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Calculate ATR
def calculate_atr(data, period=14):
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = true_range.rolling(window=period).mean()
    return atr

data['ATR'] = calculate_atr(data)

# Apply Hodrick-Prescott Filter
cycle, trend = hpfilter(data['Close'], lamb=1600)
data['Trend'] = trend
data['Cycle'] = cycle

# Calculate cyclical phase (upward or downward)
data['Cycle_Phase'] = np.where(data['Cycle'].diff() > 0, 'Up', 'Down')

# Initialize tradesheet
tradesheet = {
    'Trade_ID': [],
    'Entry_Date': [],
    'Exit_Date': [],
    'Entry_Price': [],
    'Exit_Price': [],
    'Position': [],  # 'Long' or 'Short'
    'Profit': []
}

# Strategy logic
position = None
trade_id = 1
for i in range(1, len(data)):
    # Long Entry
    if data['Trend'][i] > data['Trend'][i-1] and data['Cycle_Phase'][i] == 'Up' and data['ATR'][i] > 0.015 * data['Close'][i]:
        if position != 'Long':
            tradesheet['Trade_ID'].append(trade_id)
            tradesheet['Entry_Date'].append(data.index[i])
            tradesheet['Entry_Price'].append(data['Close'][i])
            tradesheet['Position'].append('Long')
            position = 'Long'
            trade_id += 1

    # Short Entry
    elif data['Trend'][i] < data['Trend'][i-1] and data['Cycle_Phase'][i] == 'Down' and data['ATR'][i] > 0.015 * data['Close'][i]:
        if position != 'Short':
            tradesheet['Trade_ID'].append(trade_id)
            tradesheet['Entry_Date'].append(data.index[i])
            tradesheet['Entry_Price'].append(data['Close'][i])
            tradesheet['Position'].append('Short')
            position = 'Short'
            trade_id += 1

    # Exit Conditions
    if position == 'Long':
        if data['Cycle_Phase'][i] == 'Down' or data['ATR'][i] < 0.01 * data['Close'][i]:
            tradesheet['Exit_Date'].append(data.index[i])
            tradesheet['Exit_Price'].append(data['Close'][i])
            tradesheet['Profit'].append(data['Close'][i] - tradesheet['Entry_Price'][-1])
            position = None
    elif position == 'Short':
        if data['Cycle_Phase'][i] == 'Up' or data['ATR'][i] < 0.01 * data['Close'][i]:
            tradesheet['Exit_Date'].append(data.index[i])
            tradesheet['Exit_Price'].append(data['Close'][i])
            tradesheet['Profit'].append(tradesheet['Entry_Price'][-1] - data['Close'][i])
            position = None

# Convert tradesheet to DataFrame
tradesheet_df = pd.DataFrame(tradesheet)
print(tradesheet_df)
