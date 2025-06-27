
import pandas as pd 
import numpy as np 
from api_call import * 
from brisklib import BriskLib
# from param import output
import talib as ta
import xlwings as xw
import itertools
import joblib as jb
import os
# from hmmlearn.hmm import GaussianHMM
# from sklearn.preprocessing import StandardScaler
# import plotly.express as px
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt



api_spot = ApiCall(exchange='NSE')
api_dates = ApiCall(data_type='ExpiryDates')
api_opt = ApiCall(data_type='Options')
indc = Indicators()



client_obj = BriskLib()
brisk_spot = BriskLib(EXCHANGE='NSE', DATATYPE='CASH_TV', CONNECTION_ONE_TIME=True, CLIENT=client_obj.CLIENT)
brisk_opt = BriskLib(EXCHANGE='NSE', DATATYPE='OPT', CONNECTION_ONE_TIME=True, CLIENT=client_obj.CLIENT)
brisk_dates = BriskLib(EXCHANGE='NSE', DATATYPE='OPT', GET_EXPIRY=True,  CONNECTION_ONE_TIME=True, CLIENT=client_obj.CLIENT)




ticker = 'NIFTY_50'
opt_ticker = 'NIFTY'



loaded_data = brisk_spot.get_data(symbol = ticker, start_date = '2019-01-01',
	till_today=True, resample_bool=False, resample_freq='30min', label = 'left', 
	remove_special_dates=True, time_filter=True, time_filter_start='09:15:00', time_filter_end='15:28:00')
loaded_data.rename(columns = {'Trade_date' : 'Datetime'}, inplace=True)



def get_freq_data(resample_freq, data, label='left', origin='09:15:00', last_candle_end=False, last_candle_time='15:30:00', freq_greater_than_1d=False):
	data = data.copy()
	data.set_index('Datetime', inplace=True)
	if freq_greater_than_1d:
		data = data.resample(resample_freq, label=label).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna().reset_index()
	else:
		# df = data
		anchor = data.index.normalize() + pd.Timedelta(origin)
		delta = data.index - anchor
		n_bins = (delta // pd.Timedelta(resample_freq)).astype(int)
		if label == 'left':
			new_index = anchor + n_bins * pd.Timedelta(resample_freq)
		elif label == 'right':
			new_index = anchor + (n_bins + 1) * pd.Timedelta(resample_freq)
		else:
			raise ValueError("Parameter 'label' must be either 'left' or 'right'.")
		data['resample_ts'] = new_index
		agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
		data = data.groupby('resample_ts', sort=False).agg(agg_dict).dropna().reset_index()
		data.rename(columns={'resample_ts': 'Datetime'}, inplace=True)
		if last_candle_end:
			data['Datetime'] = data['Datetime'].where(data['Datetime'].dt.time < pd.to_datetime(last_candle_time).time(), data['Datetime'].dt.date.astype(str) + ' ' + last_candle_time)
	return data

# freq = '31min'
# main_data = get_freq_data(freq, loaded_data, origin = '09:15:00', label = 'left', last_candle_end = False, last_candle_time = '15:25:00')



# change_time = main_data['Datetime'].dt.time.iloc[-1]

# for i in range(len(main_data)):
# 	row = main_data.iloc[i]

# 	# if row['Datetime'].time() == pd.to_datetime('15:15:00').time():
# 	if row['Datetime'].time() == change_time:

# 		# break
# 		stime = row['Datetime'].strftime('%Y-%m-%d') + ' ' + '14:45:00'
# 		etime = row['Datetime'].strftime('%Y-%m-%d') + ' ' + '15:23:00'

# 		ma_val = loaded_data.loc[(loaded_data['Datetime'] >= stime) & (loaded_data['Datetime'] <= etime)]['Close'].reset_index(drop=True)
# 		ma_val = ta.DEMA(ma_val, timeperiod=8)

# 		if not pd.isna(ma_val.iloc[-1]):
# 			main_data.at[i, 'Close'] = round(ma_val.iloc[-1], 2)
# 		else:
# 			raise('error')




# for date in main_data['Datetime'].dt.date.unique():
# 	# break
# 	day_data = main_data.loc[main_data['Datetime'].dt.date == date]
# 	if len(day_data) < 10:
# 		print(date)



# data = main_data.copy()
# data['KAMA'] = ta.KAMA(data['Close'], timeperiod=30)


# import pandas as pd
# import numpy as np


def zscore(src, period, ma_type='sma'):
	ma_func = getattr(ta, ma_type.upper())
	mean = ma_func(src, timeperiod=period)
	variance = ma_func((src - mean) ** 2, timeperiod=period)
	std = np.sqrt(variance)
	return (src - mean) / std



def efficiency_ratio(price: pd.Series, period: int = 14) -> pd.Series:
	"""Calculate Efficiency Ratio (ER) for a given price series."""
	change = price.diff(period).abs()
	volatility = price.diff().abs().rolling(window=period, min_periods=1).sum()
	er = (change / volatility).fillna(0)
	return er


def efficient_ema(price: pd.Series, alpha_mult: float = 3, period: int = 14, er_ratio_len: int = 5) -> pd.Series:
	er = efficiency_ratio(price, er_ratio_len)
	ema = pd.Series(index=price.index, dtype=float)
	ema.iloc[0] = price.iloc[0]
	for i in range(1, len(price)):
		# break
		alpha = (alpha_mult / (period + 1)) * er.iloc[i]
		# if alpha != 0:
		# 	break
		ema.iloc[i] = (ema.iloc[i-1] * (1 - alpha)) + (alpha * (price.iloc[i])) # - ema.iloc[i-1])
	return ema


def efficient_ema_atr(price: pd.Series, alpha_mult: float = 3, period: int = 14, er_ratio_len: int = 5,
                 atr_bool: bool = False,
                 src_bool: bool = False,
				 atr_series: pd.Series = pd.Series()) -> pd.Series:
    
    # Calculate efficiency ratio
    er = efficiency_ratio(price, er_ratio_len)

    # Initialize EMA series
    ema = pd.Series(index=price.index, dtype=float)
    ema.iloc[0] = price.iloc[0]
    
    for i in range(1, len(price)):
        # Get previous ATR value (atr_value[1] in Pine Script)
        atr_val = atr_series.iloc[i-1] if i > 0 and not pd.isna(atr_series.iloc[i-1]) else 0
        
        # Calculate diff based on src_bool
        if src_bool:
            diff = price.iloc[i] - price.iloc[i-1]
        else:
            diff = price.iloc[i] - (ema.iloc[i-1] if not pd.isna(ema.iloc[i-1]) else 0)
        
        # Calculate diff_sig
        prev_ema = ema.iloc[i-1] if not pd.isna(ema.iloc[i-1]) else 0
        prev_price = price.iloc[i-1] if i > 0 else price.iloc[i]
        
        if diff > 0 and prev_ema < prev_price:
            diff_sig = 1
        elif diff < 0 and prev_ema > prev_price:
            diff_sig = -1
        else:
            diff_sig = 0
        
        # Take absolute value of diff
        diff = abs(diff)
        
        # Calculate ATR move
        atr_move = diff - atr_val
        atr_move = max(atr_move, 0)  # atr_move > 0 ? atr_move : 0
        
        # Calculate alpha
        alpha = (alpha_mult / (period + 1)) * er.iloc[i]
        
        # Calculate adding_atr_move
        adding_atr_move = (atr_move * diff_sig) if atr_bool else 0
        
        # Calculate EMA
        prev_ema_val = ema.iloc[i-1] if not pd.isna(ema.iloc[i-1]) else 0
        ema.iloc[i] = (prev_ema_val * (1 - alpha)) + (price.iloc[i] * alpha) + adding_atr_move
    
    return ema


def rope_series(src, threshold):
	rope = np.zeros(len(src))
	rope[0] = src.iloc[0]
	for i in range(1, len(src)): 
		# break
		move = src.iloc[i] - rope[i-1]
		delta = max(abs(move) - threshold.iloc[i], 0)
		# if delta != 0:
		# 	break
		rope[i] = rope[i-1] + np.sign(move) * delta
	return pd.Series(rope, index=src.index)



def kama(price: pd.Series,
				   period: int = 10,
				   period_fast: int = 2,
				   period_slow: int = 30) -> pd.Series:

	sc_fast = 2.0 / (period_fast + 1)
	sc_slow = 2.0 / (period_slow + 1)

	# 2) Efficiency Ratio (ER)
	change     = price.diff(period).abs()
	volatility = price.diff().abs().rolling(window=period, min_periods=1).sum()
	er         = (change / volatility).fillna(0)

	# 3) Smoothing Constant (SC)
	sc = (er * (sc_fast - sc_slow) + sc_slow) ** 2

	# 4) Allocate output and seed first value at the first price bar
	kama = pd.Series(index=price.index, dtype=float)
	kama.iloc[0] = price.iloc[0]

	# 5) Recursive AMA update (matches AFL’s AMA())
	for i in range(1, len(price)):
		kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (price.iloc[i] - kama.iloc[i-1])

	return kama

# atr_len = 14
# atr_mult = 2
ts_out = None
pp_obj = None

def backtest(data, ema_window, alpha_mult, er_ratio_len, zperiod, zma_type, zthresh, src, atr_len = 14, atr_mult = 2, rsi_len = 14, dema_len = 10, atr_target = 1, atr_target_bool = False, sell_bool = True, buy_bool = True, skip_last = False):
	global ts_out, pp_obj
	data = data.copy()
	data['ATR'] = round(ta.ATR(data['High'], data['Low'], data['Close'], timeperiod=atr_len) * atr_mult, 2)
	data.dropna(inplace=True)
	data.reset_index(drop=True, inplace=True)
	# data['ROPE'] = round(rope_series(src = data['Close'], threshold = data['ATR']), 2)
	# data['ROPE_DIFF'] = data['ROPE'].diff()
	# data['EMA'] = round(ta.EMA(data['Close'], timeperiod=ema_window), 2)
	data['HLC3'] = ((data['High'] + data['Low'] + data['Close']) / 3).round(2)
	data['EMA'] = round((efficient_ema(price = data[src], period=ema_window, alpha_mult=alpha_mult, er_ratio_len=er_ratio_len)), 2)

	data['EMA_ATR'] = round((efficient_ema_atr(price = data[src], period=ema_window, alpha_mult=alpha_mult, er_ratio_len=er_ratio_len, atr_bool=True, src_bool=False, atr_series=data['ATR'])), 2)

	
	# data['EMA0'] = round((efficient_ema(price = data[src], period=ema_window, alpha_mult=alpha_mult, er_ratio_len=1)), 2)
	data['EMA0'] = ta.DEMA(data['Close'], timeperiod=dema_len)
	# data['ER'] = efficiency_ratio(data['Close'], er_ratio_len)
	# data['KAMA'] = round(kama(data['Close'], period=kperiod, period_fast=period_fast, period_slow=period_slow), 2)
	data['ZSCORE'] = round(zscore(data[src], period=zperiod, ma_type=zma_type), 2)
	# data['RSI'] = ta.RSI(data['Close'], timeperiod=rsi_len)
	# data['ZSCORE'] = round(zscore(data['RSI'], period=zperiod, ma_type=zma_type), 2)

	# data['Upper'] = round(data['ROPE'] + data['ATR'], 2)
	# data['Lower'] = round(data['ROPE'] - data['ATR'], 2)
	# data['UpperKC'] = round(data['EMA'] + data['ATR'], 2)
	# data['LowerKC'] = round(data['EMA'] - data['ATR'], 2)

	# data['STD'] = ta.STDDEV(data['Close'], timeperiod=std_len, nbdev = std_mult)
	# data['UpperSTD'] = data['EMA'] + data['STD']
	# data['LowerSTD'] = data['EMA'] - data['STD']


	data.dropna(inplace=True)
	data.reset_index(drop=True, inplace=True)
	
	last_time = data['Datetime'].dt.time.unique()[-1]

	# if testing:
	# 	data = data.loc[(data['Datetime'].dt.year >= testing_start)].copy()
	# 	data.reset_index(drop=True, inplace=True)

	# ent = Entropy(series=data['EMA_LAG'])
	# ent.check_stationarity(alpha=0.05)
	# ent.entropy(nbins=30)

	# ent = Entropy(series=data['STD'])
	# ent.check_stationarity(alpha=0.05)
	# ent.entropy(nbins=30)

	ts = {
		'EntryType' : {},
		'EntryPrice' : {}, 
		'ExitPrice' : {},
		'ExitTime' : {},
		'Bars' : {}, 
		'MaxHigh' : {}, 
		'MaxLow' : {},
		# 'ExitType' : {},
		# 'ROPE_DIFF' : {},
		'PL' : {}
	}

	position = 0

	# sell_bool = False
	# buy_bool = True
	rope_exit_activated = False
	dema_exit_activated = False
	for i in range(len(data)):

		row = data.iloc[i]

		if skip_last and row['Datetime'].time() == last_time and position == 0:
			continue
		# if position == -1 and row['ROPE_DIFF'] < 0:
		# 	rope_exit_activated = True

		# if position == 1:
		# 	if (row['Close'] / ts['EntryPrice'][entry_index]) > 1.0:

		if atr_target_bool and position != 0:
			if position == -1 and row['Close'] < atr_trailing_target and row['Close'] < row['EMA0']:
				dema_exit_activated = True
			
			if position == 1 and row['Close'] > atr_trailing_target and row['Close'] > row['EMA0']:
				dema_exit_activated = True

		if position != 0:
			ts['Bars'][entry_index] += 1
			ts['MaxHigh'][entry_index] = max(ts['MaxHigh'][entry_index], row['High'])
			ts['MaxLow'][entry_index] = min(ts['MaxLow'][entry_index], row['Low'])

		if position == -1 and (sell_bool and True 
						and (
							row['Close'] > row['EMA'] 
							# or (row['Close'] > row['EMA0'] and dema_exit_activated)
							or row['Close'] > row['EMA_ATR']
							# or (row['Close'] > row['ROPE']
		   					# 	and rope_exit_activated
							# 	)
							# or row['ZSCORE'] > -0.0
						)
						):
			
			# if row['Datetime'].time() == change_time:
			# 	# continue
			# 	row_15_25 = required_data.loc[(required_data['Datetime'].dt.date == row['Datetime'].date()) & (required_data['Datetime'].dt.time >= pd.to_datetime('15:24:00').time())]
			# 	row['Close'] = row_15_25['Close'].values[0]
			
			ts['ExitPrice'][entry_index] = row['Close']
			ts['ExitTime'][entry_index] = row['Datetime']
			# ts['ExitType'][entry_index] = 'LAG_MA_CROSS'
			ts['PL'][entry_index] = -(ts['ExitPrice'][entry_index] - ts['EntryPrice'][entry_index])
			position = 0
			rope_exit_activated = False
			dema_exit_activated = False

		if position == 1 and (buy_bool and True
						and (
							row['Close'] < row['EMA']
							# or (row['Close'] < row['EMA0'] and dema_exit_activated)
							# or row['Close'] < row['EMA0']
							# or row['Close'] < row['ROPE']
							# and row['ZSCORE'] < 0
							# or row['ZSCORE'] < -zthresh-2
						)
						):
			
			# if row['Datetime'].time() == change_time:
			# 	# continue
			# 	row_15_25 = required_data.loc[(required_data['Datetime'].dt.date == row['Datetime'].date()) & (required_data['Datetime'].dt.time >= pd.to_datetime('15:24:00').time())]
			# 	row['Close'] = row_15_25['Close'].values[0]

			ts['ExitPrice'][entry_index] = row['Close']
			ts['ExitTime'][entry_index] = row['Datetime']
			# ts['ExitType'][entry_index] = 'LAG_MA_CROSS'
			ts['PL'][entry_index] = (ts['ExitPrice'][entry_index] - ts['EntryPrice'][entry_index])
			position = 0
			dema_exit_activated = False



		if position == 0 and (buy_bool and True 
						and (
							# row['ROPE_DIFF'] > 0
							row['Close'] > row['EMA']
							# and row['Close'] > upper_range
							and row['ZSCORE'] > zthresh
							# and row['ZSCORE'] < zthresh + 2
							# and row['Close'] > row['ROPE']

							# and row['ER'] > 0.5
							# and row['Close'] > row['EMA0']
						)
						):
			# break
			
			# if row['Datetime'].time() == change_time:
			# 	# continue
			# 	row_15_25 = required_data.loc[(required_data['Datetime'].dt.date == row['Datetime'].date()) & (required_data['Datetime'].dt.time >= pd.to_datetime('15:24:00').time())]
			# 	row['Close'] = row_15_25['Close'].values[0]
			
			entry_index = (row['Datetime'])
			ts['EntryType'][entry_index] = 'Long'
			ts['EntryPrice'][entry_index] = row['Close']
			ts['Bars'][entry_index] = 0
			ts['MaxHigh'][entry_index] = row['High']
			ts['MaxLow'][entry_index] = row['Low']
			# ts['ROPE_DIFF'][entry_index] = row['ROPE_DIFF']
			atr_trailing_target = round(row['Close'] + (row['ATR'] * atr_target), 2)
			position = 1 
		
		if position == 0 and (sell_bool and True 
						and (
							# row['ROPE_DIFF'] < 0
							row['Close'] < row['EMA']
							# and row['Close'] < lower_range
							# and row['ZSCORE'] < -zthresh
							# and row['Close'] < row['ROPE']
							# and row['ER'] > 0.6
							# # and row['Close'] < row['LowerKC']
							# and row['RSI'] < 45
							# and row['RSI'] > 25
							# # and row['Close'] < row['LowerSTD']
							# and row['ROPE_DIFF'] < 0
							# and row['Close'] < row['EMA0']
							and row['Close'] < row['EMA_ATR']
						)
						):
			# break
			# if row['Datetime'].time() == change_time:
			# 	# continue
			# 	row_15_25 = required_data.loc[(required_data['Datetime'].dt.date == row['Datetime'].date()) & (required_data['Datetime'].dt.time >= pd.to_datetime('15:24:00').time())]
			# 	row['Close'] = row_15_25['Close'].values[0]
			
			entry_index = (row['Datetime'])
			ts['EntryType'][entry_index] = 'Short'
			ts['EntryPrice'][entry_index] = row['Close']
			ts['Bars'][entry_index] = 0
			ts['MaxHigh'][entry_index] = row['High']
			ts['MaxLow'][entry_index] = row['Low']
			# ts['ROPE_DIFF'][entry_index] = row['ROPE_DIFF']
			atr_trailing_target = round(row['Close'] - (row['ATR'] * atr_target), 2)
			position = -1 
		
		

	if len(ts) > 0:

		ts = pd.DataFrame(ts)
		ts.index.names = ['EntryTime']
		# ts['PL'].sum()
		ts_out =  ts.copy()
		# ts['ROPEDIFF_BINS'] = create_bins(abs(ts['ROPE_DIFF']), bin_size=5, print_bins=True).astype(str)
		pp = Performance_Profile(column_name='PL', date_col='ExitTime', print_output=True)
		output_dict = pp.analyze(ts, pl_col='PL', date_col='ExitTime')
		
		# xw.view(ts, table = False)

		ts_long = ts[ts['EntryType'] == 'Long'].copy()
		ts_short = ts[ts['EntryType'] == 'Short'].copy()

		output_dict['Long_PL'] = ts_long['PL'].sum()
		output_dict['Long_Avg'] = ts_long['PL'].mean()

		output_dict['Short_PL'] = ts_short['PL'].sum()
		output_dict['Short_Avg'] = ts_short['PL'].mean()
		pp_obj = pp
		# pp = Performance_Profile(column_name='PL', date_col='ExitTime', print_output=False)
		# min_month = pp.pivot_table(ts).drop('Total', axis = 1).drop('Total', axis = 0).agg(['idxmin', 'min']).T.sort_values(by=['min']).iloc[:5]
		# min_month['min'] = round(min_month['min'])
		# min_month['Year-Month'] = min_month['idxmin'].astype(int).astype(str) + '-' + min_month.index.astype(int).astype(str)
		# min_month.set_index('Year-Month', inplace=True)
		# min_month = min_month['min'].to_dict()
		# pivot = pp.pivot_table(ts)
		# output_dict['STD_MULT'] = std_mult
		# output_dict['EMA_WINDOW2'] = ema_window2
		# output_dict = output_dict | min_month
		output_dict['ATR_MULT'] = atr_mult
		output_dict['ATR_LEN'] = atr_len
		# output_dict['kperiod'] = kperiod
		# output_dict['kfast'] = kfast
		# # output_dict['kslow'] = kslow
		output_dict['MA_WINDOW'] = ema_window
		output_dict['ER_LEN'] = er_ratio_len
		output_dict['ALPHA_MULT'] = alpha_mult
		output_dict['ZPERIOD'] = zperiod
		output_dict['ZMA_TYPE'] = zma_type
		output_dict['ZTHRESH'] = zthresh
		output_dict['SRC'] = src
		output_dict['DEMA_LEN'] = dema_len
		output_dict['ATR_TARGET'] = atr_target
		# output_dict['STD_LEN'] = std_len
		# output_dict['STD_MULT'] = std_mult
		output_dict['RSI_LEN'] = rsi_len
		# output_dict['Chopp_len'] = chopp_len
		return output_dict



ema_window = 34
er_ratio_len = 10
alpha_mult = 2
zperiod = 11
zma_type = 'SMA'
zthresh = 1.0
src = 'HLC3'
atr_len = 18
atr_mult = 1.0
freq = '29min'



atr_target = 2.0
atr_target_bool = True
dema_len = 25
rsi_len = 29




sell_bool = True
buy_bool = True
skip_last = False



data = get_freq_data(freq, loaded_data, origin = '09:15:00', label = 'left', last_candle_end = False, last_candle_time = '15:25:00')
output_dict = backtest(data, ema_window=ema_window, alpha_mult=alpha_mult,er_ratio_len=er_ratio_len, zperiod=zperiod, zma_type=zma_type, zthresh=zthresh, src=src, atr_len=atr_len, atr_mult=atr_mult, sell_bool=sell_bool, buy_bool=buy_bool, skip_last = skip_last, dema_len=dema_len, rsi_len = rsi_len, atr_target=atr_target, atr_target_bool=atr_target_bool)



ts_out['EquityCurve'] = ts_out['PL'].cumsum()
ts_out['DD'] = ts_out['EquityCurve'] - ts_out['EquityCurve'].cummax()


for i in range(2, 11):
	ts_out['PLDEMA'] = ta.DEMA(ts_out['PL'], timeperiod=i)
	ts_out['DEC'] = np.where(ts_out['PLDEMA'].shift() < 0, 1, 0)
	temp_ts = ts_out[ts_out['DEC'] == 1].copy()
	print(f'LEN : {i}, PL : {temp_ts["PL"].sum()} MEAN : {temp_ts["PL"].mean()}')


xw.view(ts_out, table = False)

pivot = pp_obj.pivot_table(ts_out, pl_col = 'PL', date_col = 'ExitTime')

pivot.to_clipboard()



pp_obj.dd_df



def get_avg_dd_table(ts, pl_col = 'PL', date_col = 'ExitTime', spot_col = 'EntryPrice'):
	ts = ts.copy()
	ts['EquityCurve'] = ts['PL'].cumsum()
	ts['DD'] = ts['EquityCurve'] - ts['EquityCurve'].cummax()

	table = ts.groupby(ts[date_col].dt.year).agg(PL_SUM=(pl_col, 'sum'),COUNT=(pl_col, 'count'), PL_AVG=(pl_col, 'mean'),SPOT_AVG=(spot_col,'mean'),DD_MAX=('DD', 'min'))

	table['CAR/MDD'] = table['PL_SUM'] / table['DD_MAX']
	table['ExpAvg'] = table['SPOT_AVG'] * 0.0015
	return table.round(2)

ts = ts_out.copy()
table = get_avg_dd_table(ts_out, pl_col = 'PL', date_col = 'ExitTime', spot_col = 'EntryPrice')





#  ================================================================================
# SAVE TS 


ema_window = 45
er_ratio_len = 5
alpha_mult = 5
zperiod = 11
zma_type = 'SMA'
zthresh = 1.0
src = 'Close'
atr_len = 20
atr_mult = 1.5
freq = '29min'

dema_len = 24


ema_window = 35
er_ratio_len = 5
alpha_mult = 2
zperiod = 11
zma_type = 'SMA'
zthresh = 1.0
src = 'Close'
atr_len = 20
atr_mult = 1
freq = '29min'

sell_bool = True
buy_bool = True
skip_last = False


data = get_freq_data(freq, loaded_data, origin = '09:15:00', label = 'left', last_candle_end = False, last_candle_time = '15:25:00')
output_dict = backtest(data, ema_window=ema_window, alpha_mult=alpha_mult,er_ratio_len=er_ratio_len, zperiod=zperiod, zma_type=zma_type, zthresh=zthresh, src=src, atr_len=atr_len, atr_mult=atr_mult, sell_bool=sell_bool, buy_bool=buy_bool, skip_last = skip_last)


filepath = rf'D:\SYS\DIR TS\ZEMA\result_{ema_window}_{er_ratio_len}_{alpha_mult}_{zperiod}_{zma_type}_{zthresh}_{src}_{atr_len}_{atr_mult}_{freq}_for_short_Entry_DEMA24_and_Exit_zthresh_-0.0_DEMA24_RSI(10)_BASED_ZSCORE.xlsx'

ts_out.to_excel(filepath)



ts = ts_out.copy()
table = get_avg_dd_table(ts_out, pl_col = 'PL', date_col = 'ExitTime', spot_col = 'EntryPrice')




pp = Performance_Profile(column_name='PL', date_col='ExitTime', print_output=True)
output_df = pd.DataFrame([output_dict]).T

pp.update_file(filepath=filepath, dfs = [table, output_df], sheet_name='REPORT')



	
	





# ================================================================================

# KEEP THIS PARAMs


ema_window = 20
er_ratio_len = 5
alpha_mult = 2
zperiod = 11
zma_type = 'SMA'
zthresh = 1.0
src = 'Close'
freq = '28min'




ema_window = 30
er_ratio_len = 5
alpha_mult = 4
zperiod = 15
zma_type = 'SMA'
zthresh = 1.0
src = 'HLC3'
freq = '56min'





ema_window = 35
er_ratio_len = 5
alpha_mult = 4
zperiod = 11
zma_type = 'SMA'
zthresh = 1.0
src = 'Close'
atr_len = 20
atr_mult = 1.5
freq = '29min'




# ===============================================================================

#  YEAR WISE BEST 

# 2019
ema_window = 18
er_ratio_len = 6
alpha_mult = 4
zperiod = 17
zma_type = 'SMA'
zthresh = 1.0
src = 'Close'
freq = '14min'
data = get_freq_data(freq, loaded_data, origin = '09:15:00', label = 'left', last_candle_end = False, last_candle_time = '15:25:00')
output_dict = backtest(data, ema_window=ema_window, alpha_mult=alpha_mult,er_ratio_len=er_ratio_len, zperiod=zperiod, zma_type=zma_type, zthresh=zthresh, src=src)

pivot = pp_obj.pivot_table(ts_out, pl_col = 'PL', date_col = 'ExitTime')
pivot.to_clipboard()


# 2020
ema_window = 38
er_ratio_len = 10
alpha_mult = 4
zperiod = 12
zma_type = 'SMA'
zthresh = 1.0
src = 'Close'
freq = '14min'
data = get_freq_data(freq, loaded_data, origin = '09:15:00', label = 'left', last_candle_end = False, last_candle_time = '15:25:00')
output_dict = backtest(data, ema_window=ema_window, alpha_mult=alpha_mult,er_ratio_len=er_ratio_len, zperiod=zperiod, zma_type=zma_type, zthresh=zthresh, src=src)

pivot = pp_obj.pivot_table(ts_out, pl_col = 'PL', date_col = 'ExitTime')
pivot.to_clipboard()

xw.view(ts_out, table = False)

# 2021
ema_window = 10
er_ratio_len = 6
alpha_mult = 5
zperiod = 10
zma_type = 'SMA'
zthresh = 1.0
src = 'HLC3'
freq = '56min'
data = get_freq_data(freq, loaded_data, origin = '09:15:00', label = 'left', last_candle_end = False, last_candle_time = '15:25:00')
output_dict = backtest(data, ema_window=ema_window, alpha_mult=alpha_mult,er_ratio_len=er_ratio_len, zperiod=zperiod, zma_type=zma_type, zthresh=zthresh, src=src)

pivot = pp_obj.pivot_table(ts_out, pl_col = 'PL', date_col = 'ExitTime')
pivot.to_clipboard()



# 2022
ema_window = 28
er_ratio_len = 5
alpha_mult = 3
zperiod = 10
zma_type = 'SMA'
zthresh = 1.0
src = 'HLC3'
freq = '14min'
data = get_freq_data(freq, loaded_data, origin = '09:15:00', label = 'left', last_candle_end = False, last_candle_time = '15:25:00')
output_dict = backtest(data, ema_window=ema_window, alpha_mult=alpha_mult,er_ratio_len=er_ratio_len, zperiod=zperiod, zma_type=zma_type, zthresh=zthresh, src=src)

pivot = pp_obj.pivot_table(ts_out, pl_col = 'PL', date_col = 'ExitTime')
pivot.to_clipboard()


# 2023
ema_window = 34
er_ratio_len = 5
alpha_mult = 2
zperiod = 20
zma_type = 'SMA'
zthresh = 1.0
src = 'Close'
freq = '14min'
data = get_freq_data(freq, loaded_data, origin = '09:15:00', label = 'left', last_candle_end = False, last_candle_time = '15:25:00')
output_dict = backtest(data, ema_window=ema_window, alpha_mult=alpha_mult,er_ratio_len=er_ratio_len, zperiod=zperiod, zma_type=zma_type, zthresh=zthresh, src=src)

pivot = pp_obj.pivot_table(ts_out, pl_col = 'PL', date_col = 'ExitTime')
pivot.to_clipboard()


# 2024
ema_window = 26
er_ratio_len = 5
alpha_mult = 2
zperiod = 11
zma_type = 'SMA'
zthresh = 1.0
src = 'Close'
freq = '42min'
data = get_freq_data(freq, loaded_data, origin = '09:15:00', label = 'left', last_candle_end = False, last_candle_time = '15:25:00')
output_dict = backtest(data, ema_window=ema_window, alpha_mult=alpha_mult,er_ratio_len=er_ratio_len, zperiod=zperiod, zma_type=zma_type, zthresh=zthresh, src=src)

pivot = pp_obj.pivot_table(ts_out, pl_col = 'PL', date_col = 'ExitTime')
pivot.to_clipboard()



# =====================================================================================

#  DEMA OPTIMIZE 

dema_len_list = np.arange(20, 41, 1)
atr_len_list = np.arange(10, 30, 1)
atr_mult_list = np.arange(0.5, 2.5, 0.1)
rsi_len_list = np.arange(10, 30, 1)
z_len_list = np.arange(10, 30, 1)
z_mult_list = np.arange(0.5, 2.5, 0.1)
atr_target_bool = True
atr_target_list = np.arange(1, 7.5, 0.5)



comb_list = list(itertools.product(atr_target_list))
comb_list = list(itertools.product(z_len_list, z_mult_list, rsi_len_list))

comb_list = list(itertools.product(atr_len_list, atr_target_list, dema_len_list))
# len(comb_list)
print('Total Combinations:', len(comb_list))


max_pl = 0

atr_mult = 1.0

# for dema_len in dema_len_list:
# 	print(f'--------------------------------------------- {dema_len} ---------------------------------------------')
# 	output_dict = backtest(data, ema_window=ema_window, alpha_mult=alpha_mult,er_ratio_len=er_ratio_len, zperiod=zperiod, zma_type=zma_type, zthresh=zthresh, src=src, atr_len=atr_len, atr_mult=atr_mult, sell_bool=sell_bool, buy_bool=buy_bool, skip_last = skip_last, dema_len=dema_len)

# 	if output_dict['PL'] > max_pl:
# 		max_pl = output_dict['PL']
# 		max_dema_len = dema_len



results = jb.Parallel(n_jobs=10, verbose=1)(jb.delayed(backtest) (data, ema_window=ema_window, alpha_mult=alpha_mult,er_ratio_len=er_ratio_len, zperiod=zperiod, zma_type=zma_type, zthresh=zthresh, src=src, atr_len=atr_len, atr_mult=atr_mult, sell_bool=sell_bool, buy_bool=buy_bool, skip_last = skip_last, dema_len=int(dema_len), rsi_len = rsi_len) for zperiod, zthresh, rsi_len in comb_list)


results = jb.Parallel(n_jobs=10, verbose=1)(jb.delayed(backtest) (data, ema_window=ema_window, alpha_mult=alpha_mult,er_ratio_len=er_ratio_len, zperiod=zperiod, zma_type=zma_type, zthresh=zthresh, src=src, atr_len=atr_len, atr_mult=atr_mult, sell_bool=sell_bool, buy_bool=buy_bool, skip_last = skip_last, dema_len=int(dema_len), rsi_len = rsi_len, atr_target_bool=atr_target_bool, atr_target = atr_target) for atr_len, atr_target, dema_len in comb_list)





results = jb.Parallel(n_jobs=10, verbose=1)(jb.delayed(backtest) (data, ema_window=ema_window, alpha_mult=alpha_mult,er_ratio_len=er_ratio_len, zperiod=zperiod, zma_type=zma_type, zthresh=zthresh, src=src, atr_len=atr_len, atr_mult=atr_mult, sell_bool=sell_bool, buy_bool=buy_bool, skip_last = skip_last, dema_len=int(dema_len), rsi_len = rsi_len, atr_target_bool=atr_target_bool, atr_target = atr_target) for atr_target in atr_target_list)



results_df = pd.DataFrame(results)
results_df.sort_values(by = 'PL', inplace = True, ascending=False)

# save_results = results_df.copy()


#  ===============================================================================
#  OPTIMIZE 



# timeframe_list = np.arange(10, 201, 1)
# np.random.shuffle(timeframe_list)

# folder_path = r'D:\SYS\DIR\ZEMA'

# ema_window_list = np.arange(10, 50, 1)
# er_ratio_len_list = np.arange(5, 11, 1)
# alpha_mult_list = np.arange(2, 6, 1)
# zperiod_list = np.arange(10, 21, 1)
# # zma_type_list = ['SMA', 'EMA']
# zthresh_list = np.arange(1, 2.5, 0.5)
# src_list = ['Close', 'HLC3']

# comb_list = list(itertools.product(ema_window_list, er_ratio_len_list, alpha_mult_list, zperiod_list, zthresh_list, src_list))
# # len(comb_list)

# print('Total Combinations:', len(comb_list))
# print(f'TimeFrame : {timeframe_list}' )

# # timeframe_list = timeframe_list[1:]

# for t in timeframe_list:
# 	print(f'********************************************** {t}min started ***********************************************')

# 	if os.path.exists(rf'{folder_path}\{t}min_result.csv'):
# 		print(f'********************************************** {t}min done ***********************************************')
# 		continue

# 	# freq = '31min'
# 	data = get_freq_data(f'{t}min', loaded_data, origin = '09:15:00', label = 'left', last_candle_end = False, last_candle_time = '15:25:00')

# 	results = jb.Parallel(n_jobs=15, verbose = 1)(jb.delayed(backtest) (data, ema_window=ema_window, alpha_mult=alpha_mult,er_ratio_len=er_ratio_len, zperiod=zperiod, zma_type='SMA', zthresh=zthresh, src=src) for ema_window, er_ratio_len, alpha_mult, zperiod, zthresh, src in comb_list)
# 	results = [i for i in results if i is not None]

# 	results_df = pd.DataFrame(results)
# 	results_df.sort_values(by='PL', ascending=False, inplace=True)

# 	results_df.to_csv(rf'{folder_path}\{t}min_result.csv', index=False)

# 	print(f'********************************************** {t}min done ***********************************************')




from tvDatafeed import TvDatafeed, Interval

tv = TvDatafeed()

nifty_data = tv.get_hist(symbol='NIFTY', exchange='NSE',interval=Interval.in_daily,n_bars=10000).reset_index()
vix_data = tv.get_hist(symbol='INDIAVIX', exchange='NSE',interval=Interval.in_daily,n_bars=10000).reset_index()

nifty_data['CC'] = np.where(nifty_data['open'] < nifty_data['close'], 'G', 'R')
vix_data['CC'] = np.where(vix_data['open'] < vix_data['close'], 'G', 'R')


final_data = pd.merge(vix_data[['datetime', 'CC']], nifty_data[['datetime', 'CC']], on='datetime', how='left', suffixes=('_VIX', '_NIFTY'))


green_df = final_data.loc[(final_data['CC_VIX'] == 'G') & (final_data['CC_NIFTY'] == 'G')].copy()

xw.view(green_df, table = False)



# ADDED ROPE 


atr_len_list = np.arange(5, 11, 1)
atr_mult_list = np.arange(0.5, 2.5, 0.1)

comb_list = list(itertools.product(atr_len_list, atr_mult_list))
# len(comb_list)

print('Total Combinations:', len(comb_list))


for atr_len, atr_mult in comb_list: #break
	print('**********************************************', atr_len, atr_mult, '***********************************************')
	output_dict = backtest(data, ema_window=ema_window, alpha_mult=alpha_mult,er_ratio_len=er_ratio_len, zperiod=zperiod, zma_type=zma_type, zthresh=zthresh, src=src, atr_len=atr_len, atr_mult=atr_mult, sell_bool=sell_bool, buy_bool=False, skip_last = False)


















