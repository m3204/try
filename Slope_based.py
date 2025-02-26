import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load OHLC data
data = pd.read_csv("SPY_OHLC.csv")
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

# Calculate rolling regression slope and R-squared
def rolling_regression(data, window=20):
    slopes = []
    r_squared = []
    for i in range(len(data) - window):
        x = np.arange(window).reshape(-1, 1)
        y = data['Close'].iloc[i:i+window].values
        model = LinearRegression().fit(x, y)
        slopes.append(model.coef_[0])
        r_squared.append(model.score(x, y))
    # Pad with NaN for the first `window` values
    slopes = [np.nan] * window + slopes
    r_squared = [np.nan] * window + r_squared
    return slopes, r_squared

data['Slope'], data['R_Squared'] = rolling_regression(data)

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
    if data['Slope'][i] > 0 and data['R_Squared'][i] > 0.7 and data['ATR'][i] > 0.01 * data['Close'][i]:
        if position != 'Long':
            tradesheet['Trade_ID'].append(trade_id)
            tradesheet['Entry_Date'].append(data.index[i])
            tradesheet['Entry_Price'].append(data['Close'][i])
            tradesheet['Position'].append('Long')
            position = 'Long'
            trade_id += 1

    # Short Entry
    elif data['Slope'][i] < 0 and data['R_Squared'][i] > 0.7 and data['ATR'][i] > 0.01 * data['Close'][i]:
        if position != 'Short':
            tradesheet['Trade_ID'].append(trade_id)
            tradesheet['Entry_Date'].append(data.index[i])
            tradesheet['Entry_Price'].append(data['Close'][i])
            tradesheet['Position'].append('Short')
            position = 'Short'
            trade_id += 1

    # Exit Conditions
    if position == 'Long':
        if data['Slope'][i] < 0 or data['R_Squared'][i] < 0.5:
            tradesheet['Exit_Date'].append(data.index[i])
            tradesheet['Exit_Price'].append(data['Close'][i])
            tradesheet['Profit'].append(data['Close'][i] - tradesheet['Entry_Price'][-1])
            position = None
    elif position == 'Short':
        if data['Slope'][i] > 0 or data['R_Squared'][i] < 0.5:
            tradesheet['Exit_Date'].append(data.index[i])
            tradesheet['Exit_Price'].append(data['Close'][i])
            tradesheet['Profit'].append(tradesheet['Entry_Price'][-1] - data['Close'][i])
            position = None

# Convert tradesheet to DataFrame
tradesheet_df = pd.DataFrame(tradesheet)
print(tradesheet_df)
