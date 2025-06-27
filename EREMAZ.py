
import pandas as pd 
import numpy as np 
from api_call import * 
from brisklib import BriskLib
# from param import output
import talib as ta
# import xlwings as xw
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

def backtest(data, ema_window, alpha_mult, er_ratio_len, zperiod, zma_type, zthresh, src):
	# global ts_out, pp_obj
	data = data.copy()
	# data['ATR'] = round(ta.ATR(data['High'], data['Low'], data['Close'], timeperiod=atr_len) * atr_mult, 2)
	# data.dropna(inplace=True)
	# data.reset_index(drop=True, inplace=True)
	# data['ROPE'] = round(rope_series(src = data['Close'], threshold = data['ATR']), 2)
	# data['ROPE_DIFF'] = data['ROPE'].diff()
	# data['EMA'] = round(ta.EMA(data['Close'], timeperiod=ema_window), 2)
	data['HLC3'] = ((data['High'] + data['Low'] + data['Close']) / 3).round(2)
	data['EMA'] = round((efficient_ema(price = data[src], period=ema_window, alpha_mult=alpha_mult, er_ratio_len=er_ratio_len)), 2)
	# data['KAMA'] = round(kama(data['Close'], period=kperiod, period_fast=period_fast, period_slow=period_slow), 2)
	data['ZSCORE'] = round(zscore(data[src], period=zperiod, ma_type=zma_type), 2)
	# data['Upper'] = round(data['ROPE'] + data['ATR'], 2)
	# data['Lower'] = round(data['ROPE'] - data['ATR'], 2)
	# data['UpperKC'] = round(data['EMA'] + data['ATR'], 2)
	# data['LowerKC'] = round(data['EMA'] - data['ATR'], 2)

	# data['STD'] = ta.STDDEV(data['Close'], timeperiod=std_len, nbdev = std_mult)
	# data['UpperSTD'] = data['EMA'] + data['STD']
	# data['LowerSTD'] = data['EMA'] - data['STD']

	# data['RSI'] = ta.RSI(data['Close'], timeperiod=rsi_len)

	data.dropna(inplace=True)
	data.reset_index(drop=True, inplace=True)

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
		# 'ExitType' : {},
		# 'ROPE_DIFF' : {},
		'PL' : {}
	}

	position = 0

	sell_bool = True
	buy_bool = True

	for i in range(len(data)):

		row = data.iloc[i]


		if position == -1 and (sell_bool and True 
						and (
							row['Close'] > row['EMA']
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

		if position == 1 and (buy_bool and True
						and (
							row['Close'] < row['EMA']
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


		if position == 0 and (buy_bool and True 
						and (
							# row['ROPE_DIFF'] > 0
							# and row['Close'] > row['ROPE']
							row['Close'] > row['EMA']
							# and row['Close'] > upper_range
							and row['ZSCORE'] > zthresh
							# and row['Close'] > row['UpperKC']
							# and row['RSI'] > 55
							# and row['Close'] > row['UpperSTD']
							#    and row['ROPE_DIFF'] > 10
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
			# ts['ROPE_DIFF'][entry_index] = row['ROPE_DIFF']
			position = 1 
		
		if position == 0 and (sell_bool and True 
						and (
							# row['ROPE_DIFF'] < 0
							# and row['Close'] < row['ROPE']
							row['Close'] < row['EMA']
							# and row['Close'] < lower_range
							and row['ZSCORE'] < -zthresh
							# and row['Close'] < row['LowerKC']
							# and row['RSI'] < 45
							# and row['Close'] < row['LowerSTD']
							# and row['ROPE_DIFF'] < -10
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
			# ts['ROPE_DIFF'][entry_index] = row['ROPE_DIFF']
			position = -1 
		
		

	if len(ts) > 0:

		ts = pd.DataFrame(ts)
		ts.index.names = ['EntryTime']
		# ts['PL'].sum()
		# ts_out =  ts.copy()
		# ts['ROPEDIFF_BINS'] = create_bins(abs(ts['ROPE_DIFF']), bin_size=5, print_bins=True).astype(str)
		pp = Performance_Profile(column_name='PL', date_col='ExitTime', print_output=False)
		output_dict = pp.analyze(ts, pl_col='PL', date_col='ExitTime')
		
		# xw.view(ts, table = False)

		ts_long = ts[ts['EntryType'] == 'Long'].copy()
		ts_short = ts[ts['EntryType'] == 'Short'].copy()

		output_dict['Long_PL'] = ts_long['PL'].sum()
		output_dict['Long_Avg'] = ts_long['PL'].mean()

		output_dict['Short_PL'] = ts_short['PL'].sum()
		output_dict['Short_Avg'] = ts_short['PL'].mean()
		# pp_obj = pp
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
		# output_dict['ATR_MULT'] = atr_mult
		# output_dict['ATR_LEN'] = atr_len
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
		# output_dict['STD_LEN'] = std_len
		# output_dict['STD_MULT'] = std_mult
		# output_dict['RSI_LEN'] = rsi_len
		# output_dict['Chopp_len'] = chopp_len
		return output_dict



# ema_window = 21
# er_ratio_len = 3
# alpha_mult = 3
# zperiod = 20
# zma_type = 'SMA'
# zthresh = 1.5
# src = 'Close'

# # std_len = 30
# # std_mult = 0.5
# # rsi_len = 14


# freq = '13min'
# data = get_freq_data(freq, loaded_data, origin = '09:15:00', label = 'left', last_candle_end = False, last_candle_time = '15:25:00')


# output_dict = backtest(data, ema_window=ema_window, alpha_mult=alpha_mult,er_ratio_len=er_ratio_len, zperiod=zperiod, zma_type=zma_type, zthresh=zthresh, src=src)


#  ===============================================================================
#  OPTIMIZE 



timeframe_list = np.arange(10, 201, 1)
np.random.shuffle(timeframe_list)

folder_path = r'D:\SYS\DIR\ZEMA'

ema_window_list = np.arange(10, 50, 1)
er_ratio_len_list = np.arange(5, 11, 1)
alpha_mult_list = np.arange(2, 6, 1)
zperiod_list = np.arange(10, 21, 1)
# zma_type_list = ['SMA', 'EMA']
zthresh_list = np.arange(1, 2.5, 0.5)
src_list = ['Close', 'HLC3']

comb_list = list(itertools.product(ema_window_list, er_ratio_len_list, alpha_mult_list, zperiod_list, zthresh_list, src_list))
# len(comb_list)

print('Total Combinations:', len(comb_list))
print(f'TimeFrame : {timeframe_list}' )

# timeframe_list = timeframe_list[1:]

for t in timeframe_list:
	print(f'********************************************** {t}min started ***********************************************')

	if os.path.exists(rf'{folder_path}\{t}min_result.csv'):
		print(f'********************************************** {t}min done ***********************************************')
		continue

	# freq = '31min'
	data = get_freq_data(f'{t}min', loaded_data, origin = '09:15:00', label = 'left', last_candle_end = False, last_candle_time = '15:25:00')

	results = jb.Parallel(n_jobs=15, verbose = 1)(jb.delayed(backtest) (data, ema_window=ema_window, alpha_mult=alpha_mult,er_ratio_len=er_ratio_len, zperiod=zperiod, zma_type='SMA', zthresh=zthresh, src=src) for ema_window, er_ratio_len, alpha_mult, zperiod, zthresh, src in comb_list)
	results = [i for i in results if i is not None]

	results_df = pd.DataFrame(results)
	results_df.sort_values(by='PL', ascending=False, inplace=True)

	results_df.to_csv(rf'{folder_path}\{t}min_result.csv', index=False)

	print(f'********************************************** {t}min done ***********************************************')

