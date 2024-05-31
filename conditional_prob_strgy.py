
from tvDatafeed import TvDatafeed, Interval 
import numpy as np
import pandas as pd

tv = TvDatafeed(username='schauhan951', password='Schauhan@951')
data = tv.get_hist(symbol = 'BANKNIFTY', exchange='NSE', n_bars=2500)



data['OC_COL'] = np.where(data['close'] > data['open'], 'G', 'R')
data['CO_COL'] = np.where(data['close'].shift() < data['open'], 'U', 'D')

data['CO_COL'] = np.where(data['close'] < data['open'].shift(-1), 'U', 'D')


data.reset_index(inplace=True)

idx = 15
row = data.loc[idx]


ts = pd.DataFrame()
trade_open = False

back_days = 15

for idx, row in data.iterrows():
	# break
	
	if idx < back_days:
		continue 

	# break

	sample = data.query('index >= @idx-@back_days & index < @idx')
	if trade_open:
		ts.loc[(entry_time), 'Exit_Time'] = row['datetime']
		ts.loc[(entry_time), 'Exit_Price'] = row['open']
		trade_open = False

	if row['OC_COL'] == 'G':
		pu = round(len(sample.query('CO_COL == "U"')) / len(sample), 2)
		try:
			pug = round(len(sample.query('CO_COL == "U" & OC_COL == "G"')) / len(sample.query('OC_COL == "G"')), 2)
		except:
			pug = -1
		data.loc[idx, 'P(U|G)'] = pug
		data.loc[idx, 'P(U)'] = pu
		if pug >= 0.8 and pu >= 0.7:

			ts.loc[(row['datetime']), 'P(U|G)'] = pug
			ts.loc[(row['datetime']), 'P(U|R)'] = pu
			
			ts.loc[(row['datetime']), 'Entry_Price'] = row['close']
			ts.loc[(row['datetime']), 'OC_CO'] = row['OC_COL'] + '_' + row['CO_COL']
			entry_time = row['datetime']
			trade_open = True

	if row['OC_COL'] == 'R':
		p_d = round(len(sample.query('CO_COL == "D"')) / len(sample), 2)
		try:
			pdr = round(len(sample.query('CO_COL == "D" & OC_COL == "R"')) / len(sample.query('OC_COL == "R"')), 2)
		except:
			pdr = -1
		data.loc[idx, 'P(D)'] = p_d
		data.loc[idx, 'P(D|R)'] = pdr
		if pdr >= 0.8 and p_d >= 0.6:

			ts.loc[(row['datetime']), 'P(U|G)'] = pdr
			ts.loc[(row['datetime']), 'P(U|R)'] = p_d
			
			ts.loc[(row['datetime']), 'Entry_Price'] = row['close']
			ts.loc[(row['datetime']), 'OC_CO'] = row['OC_COL'] + '_' + row['CO_COL']
			entry_time = row['datetime']
			trade_open = True

	


	# data.loc[idx, 'P(U|G)'] = round((data.loc[idx, 'P(G|U)'] * data.loc[idx, 'P(U)']) / data.loc[idx, 'P(G)'], 2)
	# data.loc[idx, 'P(U|R)'] = round((data.loc[idx, 'P(R|U)'] * data.loc[idx, 'P(U)']) / data.loc[idx, 'P(R)'], 2)
	# data.loc[idx, 'P(D|G)'] = round((data.loc[idx, 'P(G|D)'] * data.loc[idx, 'P(D)']) / data.loc[idx, 'P(G)'], 2)
	# data.loc[idx, 'P(D|R)'] = round((data.loc[idx, 'P(R|D)'] * data.loc[idx, 'P(D)']) / data.loc[idx, 'P(R)'], 2)














for idx, row in data.iterrows():
	# break
	
	if idx < 15:
		continue 

	# break

	sample = data.query('index >= @idx-15 & index < @idx')

	if row['OC_COL'] == 'G':
		pass 
	try:
		data.loc[idx, 'P(G|U)'] = round(len(sample.query('CO_COL == "U" & OC_COL == "G"')) / 
								len(sample.query('CO_COL == "U"')), 2)
	except:
		data.loc[idx, 'P(G|U)'] = -1

	try:
		data.loc[idx, 'P(G|D)'] = round(len(sample.query('CO_COL == "D" & OC_COL == "G"')) / 
								len(sample.query('CO_COL == "D"')), 2)
	
	except:
		data.loc[idx, 'P(G|D)'] = -1

	try:
		data.loc[idx, 'P(R|U)'] = round(len(sample.query('CO_COL == "U" & OC_COL == "R"')) / 
								len(sample.query('CO_COL == "U"')), 2)

	except:
		data.loc[idx, 'P(R|U)'] = -1
	try:
		data.loc[idx, 'P(R|D)'] = round(len(sample.query('CO_COL == "D" & OC_COL == "R"')) / 
								len(sample.query('CO_COL == "D"')), 2)
	except:
		data.loc[idx, 'P(R|D)'] = -1

	try:
		data.loc[idx, 'P(G)'] = round(len(sample.query('OC_COL == "G"')) / len(sample), 2)

	except:
		data.loc[idx, 'P(G)'] = -1
	try:
		data.loc[idx, 'P(R)'] = round(len(sample.query('OC_COL == "R"')) / len(sample), 2)
	except:
		data.loc[idx, 'P(R)'] = -1
	
	data.loc[idx, 'P(U)'] = round(len(sample.query('CO_COL == "U"')) / len(sample), 2)

	data.loc[idx, 'P(D)'] = round(len(sample.query('CO_COL == "D"')) / len(sample), 2)

	data.loc[idx, 'P(U|G)'] = round((data.loc[idx, 'P(G|U)'] * data.loc[idx, 'P(U)']) / data.loc[idx, 'P(G)'], 2)
	data.loc[idx, 'P(U|R)'] = round((data.loc[idx, 'P(R|U)'] * data.loc[idx, 'P(U)']) / data.loc[idx, 'P(R)'], 2)
	data.loc[idx, 'P(D|G)'] = round((data.loc[idx, 'P(G|D)'] * data.loc[idx, 'P(D)']) / data.loc[idx, 'P(G)'], 2)
	data.loc[idx, 'P(D|R)'] = round((data.loc[idx, 'P(R|D)'] * data.loc[idx, 'P(D)']) / data.loc[idx, 'P(R)'], 2)


ts['PL'] = ts['Exit_Price'] - ts['Entry_Price']
ts.index.name = 'Date'
filename = 'tempo_prob_ts_bnf2.xlsx'
ts.to_excel(filename)

# ts = ts.loc[(ts.index.year >= 2020)].copy()

from metrics import Performance_Profile
import quantstats as qs

pp = Performance_Profile(column_name='PL')

points = pp.points_gained_lost(ts.reset_index())
mdd, drawdown, df = pp.drawdown_df(ts.reset_index(), want_df =True)
streak = pp.streak(ts.reset_index())
pivot = pp.pivot_table(ts.reset_index())

pp.update_file(filepath=filename, dfs = [pivot, drawdown, points, mdd, streak], sheet_name='Summary', startrow=2)
pp.update_file(filepath=filename, dfs = [data], sheet_name='Data')
pp.update_file(filepath=filename, dfs = [df], sheet_name='Capital')



qs.reports.html(returns = df['Capital_Pct'], mode = 'full', output = f'{filename.split(".")[0]}.html')








