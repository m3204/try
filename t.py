import pandas as pd 
import numpy as np
import talib as ta
from tvDatafeed import TvDatafeed, Interval
import json
from tabulate import tabulate
import tkinter as tk
from tkinter import scrolledtext
from tkinter import messagebox
from tkinter import ttk
from datetime import datetime, timedelta, time
# import sys
# sys.stdout.reconfigure(line_buffering=True)

print('Loading Data...', flush=True)


# with open('trade_open.json', 'w') as f:
#     json.dump({'Trade_Open': False, 
#                'Trade_Type' : None,
#                'EntryPrice' : 0,
#                'EntryTime' : None,
#                'ExitPrice' : 0,
#                'ExitTime' : None,
#                'PL' : 0,
#                }, f)






class PrepareData:
	def __init__(self, filepath = r"NIFTY_29min.parquet", tv_obj = None, freq = '29min', data = None):
		self.filepath = filepath
		if data is None:
			self.data = pd.read_parquet(filepath)
		else:
			self.data = data
		self.tv_obj = tv_obj
		self.freq = freq

	def get_freq_data(self, resample_freq, data, label='left', origin='09:15:00', last_candle_end=False, last_candle_time='15:30:00', freq_greater_than_1d=False):
		data = data.copy()
		# data.set_index('Datetime', inplace=True)
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

	def get_today_data(self):
		try:
			today_data = self.tv_obj.get_hist(symbol='NIFTY', interval=Interval.in_1_minute, exchange='NSE', n_bars = 375)
			if today_data is None:
				raise AttributeError('TradingView Data is not fetched, please try again later...')
			if pd.to_datetime('today').time() < pd.to_datetime('15:30:00').time() and today_data.index[-1].strftime('%H:%M') != pd.to_datetime('today').strftime('%H:%M'):
				raise TimeoutError('TradingView Data is not fetched for current minute, please try again later...')
			
			today_data = today_data.loc[(today_data.index.date == pd.Timestamp.today().date())].copy()
			if today_data.empty:
				raise AttributeError("Today's Data is not fetched, please try again later...")

			today_data = today_data.rename(columns = {'open' : 'Open', 'high' : 'High', 'low' : 'Low', 'close' : 'Close'})
			today_data.index.name = 'Datetime'
			today_data = today_data[['Open', 'High', 'Low', 'Close']].copy()
			today_data = today_data.loc[(today_data.index.time <= pd.to_datetime('15:28:00').time())].copy()
			today_data = self.get_freq_data(self.freq, today_data, label='left', origin='09:15:00')
			today_data['CloseTime'] = today_data['Datetime'] + pd.Timedelta(self.freq)

			# today_data = today_data.resample(self.freq, origin = '09:15:00', label = 'left').agg({'Open' : 'first', 'High' : 'max', 'Low' : 'min', 'Close' : 'last'})
			# today_data = today_data.reset_index()
		except AttributeError as e:
			raise(e)
		
		return today_data
	
	def get_data(self):
		self.today_data = self.get_today_data()
		data = pd.concat([self.data, self.today_data]).drop_duplicates(subset=['Datetime']).reset_index(drop=True)
		return data

	def zscore(self, src, period):
		# ma_func = getattr(ta, ma_type.upper())
		mean = ta.SMA(src, timeperiod=period)
		variance = ta.SMA((src - mean) ** 2, timeperiod=period)
		std = np.sqrt(variance)
		return (src - mean) / std

	def efficiency_ratio(self, price: pd.Series, period: int = 14) -> pd.Series:
		change = price.diff(period).abs()
		volatility = price.diff().abs().rolling(window=period, min_periods=1).sum()
		er = (change / volatility).fillna(0)
		return er


	def efficient_ema(self, price: pd.Series, alpha_mult: float = 3, period: int = 14, er_ratio_len: int = 5) -> pd.Series:
		er = self.efficiency_ratio(price, er_ratio_len)
		ema = pd.Series(index=price.index, dtype=float)
		ema.iloc[0] = price.iloc[0]
		for i in range(1, len(price)):
			# break
			alpha = (alpha_mult / (period + 1)) * er.iloc[i]
			# if alpha != 0:
			# 	break
			ema.iloc[i] = (ema.iloc[i-1] * (1 - alpha)) + (alpha * (price.iloc[i])) # - ema.iloc[i-1])
		return ema

	def calc_indc(self, data, ema_window = 45, er = 5, alpha = 5, zlen = 10, dema_len = 25):
		data['EMA'] = self.efficient_ema(data['Close'], alpha_mult = alpha, period = ema_window, er_ratio_len = er)
		data['ZSCORE'] = self.zscore(data['Close'], zlen)
		data['DEMA'] = ta.DEMA(data['Close'], dema_len)
		return data


class CreateTS(PrepareData):
	def __init__(self, filepath = r"NIFTY_29min.parquet", tv_obj = None, freq = '29min', data = None):
		super().__init__(filepath, tv_obj, freq, data = data)
		self.zup = 1.1 
		self.zdown = 2.1

	def create_ts(self):
		data = self.get_data()
		# self.save_data = data.copy()
		data = self.calc_indc(data = data)
		data = data.dropna()
		data = data.reset_index(drop=True)
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
		for i in range(len(data)):

			row = data.iloc[i]
			if position != 0:
				ts['Bars'][entry_index] += 1
				ts['MaxHigh'][entry_index] = max(ts['MaxHigh'][entry_index], row['High'])
				ts['MaxLow'][entry_index] = min(ts['MaxLow'][entry_index], row['Low'])

			if position == -1 and (row['Close'] > row['EMA'] or row['Close'] > row['DEMA']):
				ts['ExitPrice'][entry_index] = row['Close']
				ts['ExitTime'][entry_index] = row['Datetime']
				ts['PL'][entry_index] = -(ts['ExitPrice'][entry_index] - ts['EntryPrice'][entry_index])
				position = 0

			if position == 1 and (row['Close'] < row['EMA']):
				ts['ExitPrice'][entry_index] = row['Close']
				ts['ExitTime'][entry_index] = row['Datetime']
				# ts['ExitType'][entry_index] = 'LAG_MA_CROSS'
				ts['PL'][entry_index] = (ts['ExitPrice'][entry_index] - ts['EntryPrice'][entry_index])
				position = 0

			if position == 0 and (row['Close'] > row['EMA'] and row['ZSCORE'] > self.zup):
				entry_index = (row['Datetime'])
				ts['EntryType'][entry_index] = 'Long'
				ts['EntryPrice'][entry_index] = row['Close']
				ts['Bars'][entry_index] = 0
				ts['MaxHigh'][entry_index] = row['High']
				ts['MaxLow'][entry_index] = row['Low']
				position = 1 
			
			if position == 0 and (row['Close'] < row['EMA'] and row['ZSCORE'] < -self.zdown and row['Close'] < row['DEMA']):
				entry_index = (row['Datetime'])
				ts['EntryType'][entry_index] = 'Short'
				ts['EntryPrice'][entry_index] = row['Close']
				ts['Bars'][entry_index] = 0
				ts['MaxHigh'][entry_index] = row['High']
				ts['MaxLow'][entry_index] = row['Low']
				position = -1

		if len(ts['EntryPrice']) > 0:
			ts = pd.DataFrame(ts)
			ts['CUMPL'] = ts['PL'].cumsum() + 1
			ts['DD'] = ts['CUMPL'] - ts['CUMPL'].cummax()
			ts.index.names = ['EntryTime']
			ts.to_parquet('ts.parquet')
		
		# if position != 0:
		#     with open('trade_open.json', 'w') as f:
		#         json.dump({'Trade_Open': True, 
		#                 'Trade_Type' : 'SHORT' if position == -1 else 'LONG',
		#                 'EntryPrice' : ts['EntryPrice'].iloc[-1],
		#                 'EntryTime' : ts.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
		#                 'ExitPrice' : 0,
		#                 'ExitTime' : None,
		#                 'PnL' : 0,
		#                 }, f)


	
	def get_signals(self, trade_open = None):
		self.candle_times = pd.to_datetime(['09:15:00', '09:56:00', '10:37:00', '11:18:00', '11:59:00', '12:40:00', '13:21:00', '14:02:00', '14:43:00', '15:24:00'], format = '%H:%M:%S').time
		
		data = self.get_data()
		self.save_data = data.copy()
		data = self.calc_indc(data = data)

		if trade_open['Trade_Open']:
			self.curr_data = data.loc[(data['Datetime'].dt.date >= trade_open['EntryTime'].date())].copy()
		else:
			self.curr_data = data.loc[(data['Datetime'].dt.date == pd.Timestamp.today().date())].copy()
		
		try:
			
			if not trade_open['Trade_Open']:
				vix_data = self.tv_obj.get_hist(symbol='INDIAVIX', interval=Interval.in_1_minute, exchange='NSE', n_bars = 375)
				# vix_data = vix_data.loc[(vix_data.index.date == pd.Timestamp.today().date())].copy()
				vix_data = vix_data.rename(columns = {'open' : 'Open', 'high' : 'High', 'low' : 'Low', 'close' : 'Close'})
				vix_data.index.name = 'Datetime'
				vix_data = vix_data[['Open', 'High', 'Low', 'Close']].copy()
				vix_data = vix_data.loc[(vix_data.index.time <= pd.to_datetime('15:28:00').time())].copy()
				vix_data = self.get_freq_data(self.freq, vix_data, label='left', origin='09:15:00')
				vix_data['CloseTime'] = vix_data['Datetime'] + pd.Timedelta(self.freq)
				vix_data = vix_data.rename(columns = {'Close' : 'INDIAVIX'})
				self.vix_data = vix_data[['Datetime', 'INDIAVIX', 'CloseTime']]

				self.curr_data = self.curr_data.merge(self.vix_data, how = 'left', on = ['Datetime', 'CloseTime'])
			else:
				# vix_data = self.tv_obj.get_hist(symbol='INDIAVIX', interval=Interval.in_1_minute, exchange='NSE', n_bars = 3)
				if trade_open['PositionType'] == 'SYNTHETIC':
					self.curr_data['INDIAVIX'] = 0 
				if trade_open['PositionType'] == 'SPREAD':
					self.curr_data['INDIAVIX'] = 20
		except:
			self.curr_data['INDIAVIX'] = 0

		# ts = {
		#     'EntryType' : {},
		#     'EntryPrice' : {}, 
		#     'ExitPrice' : {},
		#     'ExitTime' : {},
		#     'Bars' : {}, 
		#     'MaxHigh' : {}, 
		#     'MaxLow' : {},
		#     # 'ExitType' : {},
		#     # 'ROPE_DIFF' : {},
		#     'PL' : {}
		# }

		ts = {}
		position = 0
		prev_trade = {}
		self.prev_trade_bool = False
		for i in range(len(self.curr_data)):

			row = self.curr_data.iloc[i]
			totalseconds = (row['CloseTime'] - pd.Timestamp('now')).total_seconds()
			ts['Candle_closing_Time'] = f'{int(totalseconds//60)}:{round(totalseconds%60)}'

			if row['CloseTime'] > pd.Timestamp('now') and pd.Timestamp('now') < pd.Timestamp('15:29:00'):
			# if row['CloseTime'] > pd.Timestamp('15:29:05') and pd.Timestamp('15:29:05') < pd.Timestamp('15:29:00'):
			
				if position == 0:
					ts['EntryType'] = 'No Entry'
					ts['PositionType'] = 'Chill Mode'
					ts['STRIKE'] = 0
					ts['EntryPrice'] = 0
					ts['EntryTime'] = pd.NaT
					ts['ExitPrice'] = 0
					ts['ExitTime'] = pd.NaT
					ts['MaxHigh'] = 0 
					ts['MaxLow'] = 0
					ts['Bars'] = 0
					ts['EMA'] = round(row['EMA'], 2)
					ts['DEMA'] = round(row['DEMA'], 2)
					ts['ZSCORE'] = round(row['ZSCORE'], 2)
					ts['INDIAVIX'] = row['INDIAVIX']
					ts['CURR_PRICE'] = row['Close']
					ts['CURR_TIME'] = pd.to_datetime('today').strftime('%Y-%m-%d %H:%M:%S')
					ts['PL'] = 0
				
				if position != 0:
					ts['MaxHigh'] = max(ts['MaxHigh'], row['High'])
					ts['MaxLow'] = min(ts['MaxLow'], row['Low'])
					ts['EMA'] = round(row['EMA'], 2)
					ts['DEMA'] = round(row['DEMA'], 2)
					ts['ZSCORE'] = round(row['ZSCORE'], 2)
					ts['INDIAVIX'] = row['INDIAVIX']
					ts['CURR_PRICE'] = row['Close']
					ts['CURR_TIME'] = pd.to_datetime('today').strftime('%Y-%m-%d %H:%M:%S')
					ts['PL'] = round((ts['EntryPrice'] - row['Close']), 2) if position == -1 else round((row['Close'] - ts['EntryPrice']), 2)
				continue

			if position == 0:
				ts['EntryType'] = 'No Entry'
				ts['PositionType'] = 'Chill Mode'
				ts['STRIKE'] = 0
				ts['EntryPrice'] = 0
				ts['EntryTime'] = pd.NaT
				ts['ExitPrice'] = 0
				ts['ExitTime'] = pd.NaT
				ts['MaxHigh'] = 0 
				ts['MaxLow'] = 0
				ts['Bars'] = 0
				ts['EMA'] = round(row['EMA'], 2)
				ts['DEMA'] = round(row['DEMA'], 2)
				ts['ZSCORE'] = round(row['ZSCORE'], 2)
				ts['INDIAVIX'] = row['INDIAVIX']
				ts['CURR_PRICE'] = row['Close']
				ts['CURR_TIME'] = pd.to_datetime('today').strftime('%Y-%m-%d %H:%M:%S')
				ts['PL'] = 0

			if position != 0:
				ts['Bars'] += 1
				ts['MaxHigh'] = max(ts['MaxHigh'], row['High'])
				ts['MaxLow'] = min(ts['MaxLow'], row['Low'])
				ts['EMA'] = round(row['EMA'], 2)
				ts['DEMA'] = round(row['DEMA'], 2)
				ts['ZSCORE'] = round(row['ZSCORE'], 2)
				ts['INDIAVIX'] = row['INDIAVIX']
				ts['CURR_PRICE'] = row['Close']
				ts['CURR_TIME'] = pd.to_datetime('today').strftime('%Y-%m-%d %H:%M:%S')
				ts['PL'] = round((ts['EntryPrice'] - row['Close']), 2) if position == -1 else round((row['Close'] - ts['EntryPrice']), 2)

			if position == -1 and (row['Close'] > row['EMA'] or row['Close'] > row['DEMA']):
				# break
				ts['ExitPrice'] = row['Close']
				ts['ExitTime'] = row['Datetime']
				ts['PL'] = round((ts['EntryPrice'] - ts['ExitPrice']), 2)
				position = 0

				prev_trade['EntryType'] = ts['EntryType']
				prev_trade['EntryPrice'] = ts['EntryPrice']
				prev_trade['EntryTime'] = ts['EntryTime']
				prev_trade['ExitPrice'] = ts['ExitPrice']
				prev_trade['ExitTime'] = row['CloseTime']
				prev_trade['PL'] = round(ts['PL'], 2)
				prev_trade['Bars'] = ts['Bars']
				self.prev_trade_bool = True

			if position == 1 and (row['Close'] < row['EMA']):
				ts['ExitPrice'] = row['Close']
				ts['ExitTime'] = row['Datetime']
				# ts['ExitType'] = 'LAG_MA_CROSS'
				ts['PL'] = round((ts['ExitPrice'] - ts['EntryPrice']), 2)
				position = 0

				prev_trade['EntryType'] = ts['EntryType']
				prev_trade['EntryPrice'] = ts['EntryPrice']
				prev_trade['EntryTime'] = ts['EntryTime']
				prev_trade['ExitPrice'] = ts['ExitPrice']
				prev_trade['ExitTime'] = row['CloseTime']
				prev_trade['PL'] = round(ts['PL'], 2)
				prev_trade['Bars'] = ts['Bars']
				self.prev_trade_bool = True

			if position == 0 and (row['Close'] > row['EMA'] and row['ZSCORE'] > self.zup):
				
				ts['EntryType'] = 'Long'
				ts['PositionType'] = 'SYNTHETIC' if row['INDIAVIX'] <= 15 or row['INDIAVIX'] >= 28 else 'SPREAD'
				ts['STRIKE'] = round(row['Close'], -2)
				ts['EntryPrice'] = row['Close']
				ts['EntryTime'] = row['CloseTime']
				ts['Bars'] = 0
				ts['MaxHigh'] = row['High']
				ts['MaxLow'] = row['Low']
				ts['EMA'] = round(row['EMA'], 2)
				ts['DEMA'] = round(row['DEMA'], 2)
				ts['ZSCORE'] = round(row['ZSCORE'], 2)
				ts['INDIAVIX'] = row['INDIAVIX']
				ts['ExitPrice'] = 0
				ts['ExitTime'] = pd.NaT
				ts['PL'] = 0
				position = 1
				self.prev_trade_bool = False
			
			if position == 0 and (row['Close'] < row['EMA'] and row['ZSCORE'] < -self.zdown and row['Close'] < row['DEMA']):
				# break
				ts['EntryType'] = 'Short'
				ts['PositionType'] = 'SYNTHETIC' if row['INDIAVIX'] <= 15 or row['INDIAVIX'] >= 28 else 'SPREAD'
				ts['STRIKE'] = round(row['Close'], -2)
				ts['EntryPrice'] = row['Close']
				ts['EntryTime'] = row['CloseTime']
				ts['Bars'] = 0
				ts['MaxHigh'] = row['High']
				ts['MaxLow'] = row['Low']
				ts['EMA'] = round(row['EMA'], 2)
				ts['DEMA'] = round(row['DEMA'], 2)
				ts['ZSCORE'] = round(row['ZSCORE'], 2)
				ts['INDIAVIX'] = row['INDIAVIX']
				ts['ExitPrice'] = 0
				ts['ExitTime'] = pd.NaT
				ts['PL'] = 0
				position = -1
				self.prev_trade_bool = False
		# if len(ts) > 0:
		# 	table = [(k, v) for k, v in ts.items()]
		# 	print(tabulate(table, headers=["Metric", "Value"], tablefmt="fancy_grid"))

			# output_text.config(state='normal')
			# output_text.delete('1.0', tk.END)
			# output_text.insert(tk.END, output)
			# output_text.config(state='disabled')

		if position == 0:
			if trade_open['Trade_Open']:
				trade_open['Trade_Open'] = False
				trade_open['Trade_Type'] = 'No Entry'
				trade_open['PositionType'] = 'Chill Mode'
				trade_open['EntryPrice'] = 0
				trade_open['EntryTime'] = None
				trade_open['ExitPrice'] = 0
				trade_open['ExitTime'] = None
				trade_open['PL'] = 0

				with open('trade_open.json', 'w') as f:
					json.dump(trade_open, f)
			else:
				trade_open['Trade_Open'] = False
		
		if position != 0:
			if not trade_open['Trade_Open']:
				trade_open['Trade_Open'] = True
				trade_open['Trade_Type'] = ts['EntryType']
				trade_open['PositionType'] = ts['PositionType']
				trade_open['EntryPrice'] = ts['EntryPrice']
				trade_open['EntryTime'] = ts['EntryTime'].strftime('%Y-%m-%d %H:%M:%S')
				trade_open['ExitPrice'] = 0
				trade_open['ExitTime'] = None
				trade_open['PL'] = round(ts['PL'], 2)
				with open('trade_open.json', 'w') as f:
					json.dump(trade_open, f)

			else:
				trade_open['Trade_Type'] = True

		if pd.Timestamp.today().time() > pd.to_datetime('15:30:00').time():
			self.save_data.to_parquet(self.filepath)
			# self.create_ts()

		return ts, prev_trade





# Load your data (make sure these files exist or adjust paths)
with open('trade_open.json', 'r') as f:
	trade_open = json.load(f)
	trade_open['EntryTime'] = pd.to_datetime(trade_open['EntryTime'])
tv_obj = TvDatafeed()
data = pd.read_parquet('NIFTY_29min.parquet')

# prev_entry_type = None

if trade_open['Trade_Type'] == 'Long':
	prev_entry_type = 'Long'
elif trade_open['Trade_Type'] == 'Short':
	prev_entry_type = 'Short'
else:
	prev_entry_type = None

# Simple run counter to track runs
run_counter = 0




def show_alert(message):
	alert_window = tk.Toplevel(root)
	alert_window.title("Signal Change Alert")
	alert_window.geometry("300x100")
	alert_window.transient(root)  # Ties it to the main window
	label = tk.Label(alert_window, text=message, font=("Arial", 14, "bold"), fg="red")
	label.pack(pady=10)
	ok_button = tk.Button(alert_window, text="OK", command=alert_window.destroy)
	ok_button.pack(pady=10)
	alert_window.lift()  # Bring to front
	alert_window.focus_force()  # Focus on this window
	try:
		import winsound
		winsound.Beep(751, 3000)
		 # 1000 Hz, 500 ms beep
	except ImportError:
		pass  # Ignore if sound isn't available (non-Windows platforms)


def run_analysis(tv_obj, data):
	global prev_entry_type, run_counter
	run_counter += 1
	
	freq = '29min'
	self = CreateTS(tv_obj=tv_obj, freq=freq, data=data)
	with open('trade_open.json', 'r') as f:
		trade_open = json.load(f)
		trade_open['EntryTime'] = pd.to_datetime(trade_open['EntryTime'])
	try:
		i = 0
		while True:
			try:
				result, prev_trade_details = self.get_signals(trade_open=trade_open)
				break
			except Exception as e:
				i += 1
				if i > 5: break
				else: continue

		current_entry_type = result['EntryType']
		if prev_entry_type is not None and current_entry_type != prev_entry_type:
			show_alert(f"Signal changed to {current_entry_type} from {prev_entry_type}")
		prev_entry_type = current_entry_type

		# Prepare new run output
		output_lines = []
		timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		
		# Add separator for new run (except for first run)
		if run_counter > 1:
			output_lines.append("\n" + "="*100)
			output_lines.append("="*100 + "\n")
		
		# Add current signal
		output_lines.append(f"📈 SIGNAL RUN #{run_counter} - {timestamp}\n")
		table = [(k, v) for k, v in result.items()]
		output_lines.append(tabulate(table, headers=["Metric", "Value"], tablefmt="fancy_grid"))

		# Add previous trade if available
		if self.prev_trade_bool:
			output_lines.append("\n💼 Previous Trade Details\n")
			prev_table = [(k, v) for k, v in prev_trade_details.items()]
			output_lines.append(tabulate(prev_table, headers=["Metric", "Value"], tablefmt="fancy_grid"))

		new_output = "\n".join(output_lines)

		# Instead of replacing content, append to existing content
		output_text.config(state='normal')
		
		# Insert new content at the beginning (so latest is at top)
		output_text.insert('1.0', new_output)
		
		# Find and color the EntryType in the newly added content
		lines = new_output.splitlines()
		entry_type_line = None
		for line_index, line in enumerate(lines):
			if "EntryType" in line:
				entry_type_line = line_index
				entry_line = line
				break

		if entry_type_line is not None:
			parts = entry_line.split("│")
			if len(parts) >= 3:
				value_part = parts[2].strip()
				value_start_in_line = entry_line.find(value_part)
				char_start = value_start_in_line
				char_end = char_start + len(value_part)

			line_start = f"{entry_type_line + 1}.{char_start}"
			line_end = f"{entry_type_line + 1}.{char_end}"
			try:
				entry_value = ''.join(result['EntryType'].split()).lower()
			except IndexError:
				entry_value = ""

			tag_name = f"entry_{run_counter}"
			if entry_value == "long":
				output_text.tag_add(tag_name, line_start, line_end)
				output_text.tag_config(tag_name, foreground="green", font=("Courier New", 12, "bold"))
			elif entry_value == "short":
				output_text.tag_add(tag_name, line_start, line_end)
				output_text.tag_config(tag_name, foreground="red", font=("Courier New", 12, "bold"))
			elif entry_value == 'noentry':
				output_text.tag_add(tag_name, line_start, line_end)
				output_text.tag_config(tag_name, foreground="white", font=("Courier New", 12, "bold"))

		output_text.config(state='disabled')
		# Auto-scroll to top to show latest signal
		output_text.see("1.0")

	except Exception as e:
		output_text.config(state='normal')
		output_text.insert('1.0', f"❌ Error running get_signals (Run #{run_counter}): {e}\n" + "="*80 + "\n")
		output_text.config(state='disabled')

def clear_history():
	"""Clear all the text content"""
	global run_counter
	run_counter = 0
	output_text.config(state='normal')
	output_text.delete('1.0', tk.END)
	output_text.insert(tk.END, "📝 History cleared. Ready for new runs.")
	output_text.config(state='disabled')


# Rest of your existing code remains the same...
scheduled_job_id = None
next_run_time = None
countdown_job_id = None

def update_countdown_and_progress(user_input):
	global countdown_job_id
	if next_run_time is None:
		return

	now = datetime.now()
	remaining = next_run_time - now
	total_seconds = user_input * 60
	seconds_left = max(0, int(remaining.total_seconds()))

	mins, secs = divmod(seconds_left, 60)
	countdown_label.config(text=f"Next run in: {mins:02d}m {secs:02d}s")

	progress_value = ((total_seconds - seconds_left) / total_seconds) * 100
	progress_bar['value'] = progress_value

	countdown_job_id = root.after(1000, update_countdown_and_progress, user_input)

def schedule_next_run(tv_obj, data, user_input=29):
	global scheduled_job_id, next_run_time
	now = datetime.now()
	start_time = datetime.combine(now.date(), time(9, 15))
	interval = timedelta(minutes=user_input)
	print(user_input, type(user_input))
	
	T_next = start_time + interval
	while T_next <= now:
		T_next += interval
		
	next_run_time = T_next
	delay_seconds = (T_next - now).total_seconds() + 1
	delay_ms = int(delay_seconds * 1000)

	if scheduled_job_id is not None:
		root.after_cancel(scheduled_job_id)

	scheduled_job_id = root.after(delay_ms, run_analysis_and_schedule, tv_obj, data, user_input)
	update_countdown_and_progress(user_input)

def run_analysis_and_schedule(tv_obj, data, user_input=29):
	run_analysis(tv_obj, data)
	schedule_next_run(tv_obj, data, user_input)


def on_enter(event=None, tv_obj=None, data=None):
	run_analysis(tv_obj, data)

def on_submit_interval(tv_obj=None, data=None):
	try:
		user_input = int(user_input_var.get())
		schedule_next_run(tv_obj = tv_obj, data = data, user_input=user_input)
	except ValueError:
		output_text.config(state='normal')
		output_text.insert('end', f"❌ Please enter a valid integer for interval.\n")
		output_text.config(state='disabled')

# GUI setup
root = tk.Tk()
root.title("Signal Viewer")
root.geometry("1200x800")  # Made slightly larger for history display
root.configure(bg="#2e2e2e")

# Top frame for controls
top_frame = tk.Frame(root, bg="#2e2e2e")
top_frame.pack(fill='x', pady=10)

user_input_var = tk.StringVar()
user_input_var.set("29")

# First row of controls
control_frame1 = tk.Frame(top_frame, bg="#2e2e2e")
control_frame1.pack(fill='x', pady=5)

input_label = tk.Label(control_frame1, text="Update Interval:", bg="#2e2e2e", fg="white", font=("Arial", 12))
input_label.pack(side='left', padx=(20, 5))

input_entry = tk.Entry(control_frame1, textvariable=user_input_var, font=("Arial", 12), width=10)
input_entry.pack(side='left', padx=(0, 10))

submit_button = tk.Button(
	control_frame1,
	text="✅ ",
	command=lambda: on_submit_interval(tv_obj, data),
	font=("Arial", 12),
	bg="#007bff",
	fg="white",
	padx=10,
	pady=4
)
submit_button.pack(side='left', padx=(0, 10))

# Run button
run_button = tk.Button(
	control_frame1,
	text="▶ Run Signal",
	command=lambda: run_analysis(tv_obj, data),
	font=("Arial", 14, "bold"),
	bg="#28a745",
	fg="white",
	activebackground="#218838",
	activeforeground="white",
	padx=20,
	pady=6,
	relief="raised",
	bd=3
)
run_button.pack(side='left')

# Second row of controls for history management
control_frame2 = tk.Frame(top_frame, bg="#2e2e2e")
control_frame2.pack(fill='x', pady=5)

history_label = tk.Label(control_frame2, text="History Controls:", bg="#2e2e2e", fg="white", font=("Arial", 12))
history_label.pack(side='left', padx=(20, 10))

clear_button = tk.Button(
	control_frame2,
	text="🗑️ Clear History",
	command=clear_history,
	font=("Arial", 10),
	bg="#dc3545",
	fg="white",
	padx=10,
	pady=4
)
clear_button.pack(side='left', padx=(0, 10))

# History info label
history_info_label = tk.Label(control_frame2, text="", bg="#2e2e2e", fg="yellow", font=("Arial", 10))
history_info_label.pack(side='right', padx=(0, 20))

# Function to update run info
def update_run_info():
	history_info_label.config(text=f"📊 Total runs: {run_counter}")
	root.after(1000, update_run_info)

update_run_info()

# Countdown and Progress Bar Frame
progress_frame = tk.Frame(root, bg="#2e2e2e")
progress_frame.pack(fill='x', pady=10)

countdown_label = tk.Label(progress_frame, text="Next run in: --", font=("Arial", 12), bg="#2e2e2e", fg="white")
countdown_label.pack(pady=5)

progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", length=400, mode="determinate")
progress_bar.pack()

# Main content area with scrollbar
content_frame = tk.Frame(root, bg="#2e2e2e", padx=10, pady=10)
content_frame.pack(fill='both', expand=True)

# Create scrollable text widget
text_frame = tk.Frame(content_frame, bg="#2e2e2e")
text_frame.pack(fill='both', expand=True)

output_text = tk.Text(
	text_frame,
	width=140,
	height=30,
	font=("Consolas", 10),
	wrap=tk.NONE,
	bg="#2e2e2e",
	fg="white",
	relief="sunken",
	bd=2
)

# Add scrollbars
v_scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=output_text.yview)
h_scrollbar = ttk.Scrollbar(text_frame, orient="horizontal", command=output_text.xview)
output_text.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

# Pack scrollbars and text widget
output_text.pack(side="left", fill="both", expand=True)
v_scrollbar.pack(side="right", fill="y")
h_scrollbar.pack(side="bottom", fill="x")

output_text.config(state='disabled')

# Bind Enter key to trigger run
root.bind('<Return>', lambda event: on_enter(event, tv_obj, data))

root.mainloop()







# def show_alert(message):
# 	alert_window = tk.Toplevel(root)
# 	alert_window.title("Signal Change Alert")
# 	alert_window.geometry("300x100")
# 	alert_window.transient(root)  # Ties it to the main window
# 	label = tk.Label(alert_window, text=message, font=("Arial", 14, "bold"), fg="red")
# 	label.pack(pady=10)
# 	ok_button = tk.Button(alert_window, text="OK", command=alert_window.destroy)
# 	ok_button.pack(pady=10)
# 	alert_window.lift()  # Bring to front
# 	alert_window.focus_force()  # Focus on this window
# 	try:
# 		import winsound
# 		winsound.Beep(751, 3000)
# 		 # 1000 Hz, 500 ms beep
# 	except ImportError:
# 		pass  # Ignore if sound isn't available (non-Windows platforms)

# prev_entry_type = None
# def run_analysis(tv_obj, trade_open, data):
# 	global prev_entry_type
# 	# with open('trade_open.json', 'r') as f:
# 	# 	trade_open = json.load(f)
# 	# 	trade_open['EntryTime'] = pd.to_datetime(trade_open['EntryTime'])
# 	# tv_obj = TvDatafeed()
# 	freq = '29min'
# 	self = CreateTS(tv_obj=tv_obj, freq=freq, data=data)
# 	try:
# 		# result, prev_trade_details = self.get_signals(trade_open=trade_open)

# 		# try:
# 		# 	result, prev_trade_details = self.get_signals(trade_open=trade_open)
# 		# except Exception as e:
# 		# 	result, prev_trade_details = self.get_signals(trade_open=trade_open)
# 		i = 0
# 		while True:
# 			try:
# 				result, prev_trade_details = self.get_signals(trade_open=trade_open)
# 				break
# 			except Exception as e:
# 				i += 1
# 				if i > 5: break
# 				else: continue

# 		# result['EntryType'] = 'Long'
# 		output_lines = []

# 		current_entry_type = result['EntryType']
# 		if prev_entry_type is not None and current_entry_type != prev_entry_type:
# 			show_alert(f"Signal changed to {current_entry_type} from {prev_entry_type}")
# 		prev_entry_type = current_entry_type

# 		# Add current signal
# 		output_lines.append("📈 Current Signal\n")
# 		table = [(k, v) for k, v in result.items()]
# 		output_lines.append(tabulate(table, headers=["Metric", "Value"], tablefmt="fancy_grid"))

# 		# Add previous trade if available
# 		if self.prev_trade_bool:
# 			output_lines.append("\n💼 Previous Trade Details\n")
# 			prev_table = [(k, v) for k, v in prev_trade_details.items()]
# 			output_lines.append(tabulate(prev_table, headers=["Metric", "Value"], tablefmt="fancy_grid"))

# 		full_output = "\n".join(output_lines)

# 		output_text.config(state='normal')
# 		output_text.delete('1.0', tk.END)
# 		output_text.insert(tk.END, full_output)
		
# 		entry_type_line = None
# 		for line_index, line in enumerate(full_output.splitlines()):
# 			# Look for the EntryType in the line containing the metric
# 			if "EntryType" in line:
# 				entry_type_line = line_index
# 				entry_line = line
# 				break

# 		if entry_type_line is not None:

# 			parts = entry_line.split("│")
# 			if len(parts) >= 3:
# 				# Find where the value starts in the line
# 				value_part = parts[2].strip()
# 				value_start_in_line = line.find(value_part)
				
# 				# Calculate character positions
# 				char_start = value_start_in_line
# 				char_end = char_start + len(value_part)

# 			line_start = f"{entry_type_line + 1}.{char_start}"  # +2 to adjust for header + border
# 			line_end = f"{entry_type_line + 1}.{char_end}"
# 			try:
# 				# entry_value = content.split("│")[2].strip().lower().strip()
# 				entry_value = ''.join(result['EntryType'].split()).lower()
# 			except IndexError:
# 				entry_value = ""

# 			if entry_value == "long":
# 				output_text.tag_add("entry_green", line_start, line_end)
# 				output_text.tag_config("entry_green", foreground="green", font=("Courier New", 12, "bold"))
# 			elif entry_value == "short":
# 				output_text.tag_add("entry_red", line_start, line_end)
# 				output_text.tag_config("entry_red", foreground="red", font=("Courier New", 12, "bold"))
# 			elif entry_value == 'noentry':
# 				output_text.tag_add("entry_white", line_start, line_end)
# 				output_text.tag_config("entry_white", foreground="white", font=("Courier New", 12, "bold"))

# 		output_text.config(state='disabled')

# 	except Exception as e:
# 		output_text.config(state='normal')
# 		output_text.delete('1.0', tk.END)
# 		output_text.insert(tk.END, f"Error running get_signals: {e}")
# 		output_text.config(state='disabled')


# from datetime import datetime, timedelta, time
# scheduled_job_id = None
# next_run_time = None
# countdown_job_id = None

# def update_countdown_and_progress(user_input):
# 	global countdown_job_id
# 	if next_run_time is None:
# 		return

# 	now = datetime.now()
# 	remaining = next_run_time - now
# 	total_seconds = user_input * 60
# 	seconds_left = max(0, int(remaining.total_seconds()))

# 	# Update label
# 	mins, secs = divmod(seconds_left, 60)
# 	countdown_label.config(text=f"Next run in: {mins:02d}m {secs:02d}s")

# 	# Update progress bar
# 	progress_value = ((total_seconds - seconds_left) / total_seconds) * 100
# 	progress_bar['value'] = progress_value

# 	# Schedule next countdown update
# 	countdown_job_id = root.after(1000, update_countdown_and_progress, user_input)


# def schedule_next_run(tv_obj, trade_open, data, user_input=29):
# 	global scheduled_job_id, next_run_time
# 	now = datetime.now()
# 	start_time = datetime.combine(now.date(), time(9, 15))
# 	interval = timedelta(minutes=user_input)
# 	print(user_input, type(user_input))
# 	# Start from the first candle close after 09:15
# 	T_next = start_time + interval
# 	# print(T_next)
# 	while T_next <= now:
# 		T_next += interval
# 		# print(T_next)
# 	next_run_time = T_next  # save for progress bar
# 	delay_seconds = (T_next - now).total_seconds() + 2
# 	delay_ms = int(delay_seconds * 1000)

# 	 # Cancel previously scheduled job if any
# 	if scheduled_job_id is not None:
# 		root.after_cancel(scheduled_job_id)

# 	# Schedule and store new job id
# 	scheduled_job_id = root.after(delay_ms, run_analysis_and_schedule, tv_obj, trade_open, data, user_input)

# 	# root.after(delay_ms, run_analysis_and_schedule, tv_obj, trade_open, data, user_input)
# 	# Start countdown updates
# 	update_countdown_and_progress(user_input)


# def run_analysis_and_schedule(tv_obj, trade_open, data, user_input=29):
# 	run_analysis(tv_obj, trade_open, data)
# 	schedule_next_run(tv_obj, trade_open, data, user_input)


# with open('trade_open.json', 'r') as f:
# 	trade_open = json.load(f)
# 	trade_open['EntryTime'] = pd.to_datetime(trade_open['EntryTime'])
# tv_obj = TvDatafeed()
# data = pd.read_parquet('NIFTY_29min.parquet')

# def on_enter(event=None, tv_obj=None, trade_open=None, data=None):
# 	run_analysis(tv_obj, trade_open, data)

# # GUI setup
# # root = tk.Tk()
# # root.title("Signal Viewer (No Scroll)")
# # root.geometry("1100x700")  # Wider window to show tabulated text properly
# # root.configure(bg="gray30", )

# # run_button = tk.Button(root, text="▶ Run Signal", command=lambda: run_analysis(tv_obj, trade_open, data), font=("Arial", 14), bg="green", fg="white")
# # run_button.pack(pady=10)

# # # Regular Text widget (no scroll)
# # output_text = tk.Text(root, width=140, height=140, font=("Courier New", 10), wrap=tk.NONE)
# # output_text.pack(padx=10, pady=10)
# # output_text.config(state='disabled')



# # # Optional: auto run on Enter key
# # # root.bind('<Return>', on_enter)
# # root.bind('<Return>',  lambda event: on_enter(event, tv_obj, trade_open, data))

# # schedule_next_run(tv_obj, trade_open, data)

# # root.mainloop()


# def on_submit_interval(tv_obj=None, trade_open=None, data=None):
# 	try:
# 		user_input = int(user_input_var.get())
# 		schedule_next_run(tv_obj = tv_obj, trade_open = trade_open, data = data, user_input=user_input)
# 	except ValueError:
# 		output_text.config(state='normal')
# 		output_text.insert('end', f"❌ Please enter a valid integer for interval.\n")
# 		output_text.config(state='disabled')

# root = tk.Tk()
# root.title("Signal Viewer")
# root.geometry("1100x700")
# root.configure(bg="#2e2e2e")  # Dark background

# # Top frame for button
# top_frame = tk.Frame(root, bg="#2e2e2e")
# top_frame.pack(fill='x', pady=10)


# user_input_var = tk.StringVar()
# user_input_var.set("29")  # default interval

# input_label = tk.Label(top_frame, text="Enter an Time Interval to Run:", bg="#2e2e2e", fg="white", font=("Arial", 12))
# input_label.pack(side='left', padx=(20, 5))

# input_entry = tk.Entry(top_frame, textvariable=user_input_var, font=("Arial", 12), width=10)
# input_entry.pack(side='left', padx=(0, 10))

# submit_button = tk.Button(
# 	top_frame,
# 	text="✅ Submit Interval",
# 	command=lambda: on_submit_interval(tv_obj, trade_open, data),
# 	font=("Arial", 12),
# 	bg="#007bff",
# 	fg="white",
# 	padx=10,
# 	pady=4
# )
# submit_button.pack(side='left')

# # user_input = int(user_input_var.get())

# # Run button
# run_button = tk.Button(
# 	top_frame,
# 	text="▶ Run Signal",
# 	command=lambda: run_analysis(tv_obj, trade_open, data),
# 	font=("Arial", 14, "bold"),
# 	bg="#28a745",
# 	fg="white",
# 	activebackground="#218838",
# 	activeforeground="white",
# 	padx=20,
# 	pady=6,
# 	relief="raised",
# 	bd=3
# )
# run_button.pack()


# # Countdown and Progress Bar Frame
# progress_frame = tk.Frame(root, bg="#2e2e2e")
# progress_frame.pack(fill='x', pady=10)

# countdown_label = tk.Label(progress_frame, text="Next run in: --", font=("Arial", 12), bg="#2e2e2e", fg="white")
# countdown_label.pack(pady=5)

# progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", length=400, mode="determinate")
# progress_bar.pack()

# # Main content area with frame for styling
# content_frame = tk.Frame(root, bg="#2e2e2e", padx=10, pady=10)
# content_frame.pack(fill='both', expand=True)

# # Output text area
# output_text = tk.Text(
# 	content_frame,
# 	width=140,
# 	height=30,
# 	font=("Consolas", 11),
# 	wrap=tk.NONE,
# 	bg="#2e2e2e",
# 	fg="white",
# 	relief="sunken",
# 	bd=2
# )
# output_text.pack(fill='both', expand=True)
# output_text.config(state='disabled')

# # Bind Enter key to trigger run
# root.bind('<Return>', lambda event: on_enter(event, tv_obj, trade_open, data))

# # Optional auto-run scheduler
# # schedule_next_run(tv_obj, trade_open, data, user_input = user_input)

# root.mainloop()





# self.curr_data = self.data.copy()




# self.curr_data = self.curr_data.loc[(self.curr_data['Datetime'] >= pd.to_datetime('2025-07-07 10:43:00')) & ((self.curr_data['Datetime'].dt.date == pd.to_datetime('2025-07-07').date()))]



	
	
		

		


		



		

		

# with open('trade_open.json', 'r') as f:
# 	trade_open = json.load(f)
# 	trade_open['EntryTime'] = pd.to_datetime(trade_open['EntryTime'])

# freq = '29min'
# tv_obj = TvDatafeed()
# self = CreateTS(tv_obj=tv_obj, freq = freq)


# self.get_signals(trade_open=trade_open)



# self.curr_data


# # Create GUI
# root = tk.Tk()
# root.title("Result Viewer")
# root.geometry("700x500")
# root.configure(bg="white")

# # Button
# run_button = tk.Button(root, text="▶ Run Code", command=run_analysis, font=("Arial", 14), bg="green", fg="white")
# run_button.pack(pady=10)

# # Scrollable output area
# output_text = scrolledtext.ScrolledText(root, width=80, height=25, font=("Courier", 10))
# output_text.pack(padx=10, pady=10)
# output_text.config(state='disabled')

# # Bind Enter key
# root.bind('<Return>', on_enter)

# root.mainloop()



# self.create_ts()

# self.get_data()
# self.calc_indc()

# self.data 
























































#  ========================================================================================

#  SAVE DATA 


# import pandas as pd 
# import numpy as np 
# from api_call import * 
# from brisklib import BriskLib
# # from param import output
# import talib as ta
# import xlwings as xw
# import itertools
# import joblib as jb
# import os
# # from hmmlearn.hmm import GaussianHMM
# # from sklearn.preprocessing import StandardScaler
# # import plotly.express as px
# # from sklearn.cluster import KMeans
# # import matplotlib.pyplot as plt



# api_spot = ApiCall(exchange='NSE')
# api_dates = ApiCall(data_type='ExpiryDates')
# api_opt = ApiCall(data_type='Options')
# indc = Indicators()



# client_obj = BriskLib()
# brisk_spot = BriskLib(EXCHANGE='NSE', DATATYPE='CASH_TV', CONNECTION_ONE_TIME=True, CLIENT=client_obj.CLIENT)
# brisk_opt = BriskLib(EXCHANGE='NSE', DATATYPE='OPT', CONNECTION_ONE_TIME=True, CLIENT=client_obj.CLIENT)
# brisk_dates = BriskLib(EXCHANGE='NSE', DATATYPE='OPT', GET_EXPIRY=True,  CONNECTION_ONE_TIME=True, CLIENT=client_obj.CLIENT)

# ticker = 'NIFTY_50'
# opt_ticker = 'NIFTY'

# # ticker = 'NIFTY_BANK'
# # opt_ticker = 'BANKNIFTY'

# loaded_data = brisk_spot.get_data(symbol = ticker, start_date = '2019-01-01',
# 	till_today=True, resample_bool=False, resample_freq='30min', label = 'left', 
# 	remove_special_dates=True, time_filter=True, time_filter_start='09:15:00', time_filter_end='15:28:00')
# loaded_data.rename(columns = {'Trade_date' : 'Datetime'}, inplace=True)


# def get_freq_data(resample_freq, data, label='left', origin='09:15:00', last_candle_end=False, last_candle_time='15:30:00', freq_greater_than_1d=False):
# 	data = data.copy()
# 	data.set_index('Datetime', inplace=True)
# 	if freq_greater_than_1d:
# 		data = data.resample(resample_freq, label=label).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna().reset_index()
# 	else:
# 		# df = data
# 		anchor = data.index.normalize() + pd.Timedelta(origin)
# 		delta = data.index - anchor
# 		n_bins = (delta // pd.Timedelta(resample_freq)).astype(int)
# 		if label == 'left':
# 			new_index = anchor + n_bins * pd.Timedelta(resample_freq)
# 		elif label == 'right':
# 			new_index = anchor + (n_bins + 1) * pd.Timedelta(resample_freq)
# 		else:
# 			raise ValueError("Parameter 'label' must be either 'left' or 'right'.")
# 		data['resample_ts'] = new_index
# 		agg_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
# 		data = data.groupby('resample_ts', sort=False).agg(agg_dict).dropna().reset_index()
# 		data.rename(columns={'resample_ts': 'Datetime'}, inplace=True)
# 		if last_candle_end:
# 			data['Datetime'] = data['Datetime'].where(data['Datetime'].dt.time < pd.to_datetime(last_candle_time).time(), data['Datetime'].dt.date.astype(str) + ' ' + last_candle_time)
# 	return data


# freq = '29min'
# data = get_freq_data(freq, loaded_data, origin = '09:15:00', label = 'left', last_candle_end = False, last_candle_time = '15:25:00')

# data['CloseTime'] = data['Datetime'] + pd.Timedelta('29min')

# data.to_parquet(r'C:\Python\Research\trade_execute\NF_29min_EEMA_ZSCORE_DEMA\NIFTY_29min.parquet', index=False)













