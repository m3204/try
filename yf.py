import yfinance as yf

stock_list = pd.DataFrame(os.listdir(r"D:\Data\Stock_SPOT\Final_Adjustment2"), columns = ['Ticker'])

diff_dict = pd.DataFrame()

in_out = 'inner'

for ticker in stock_list['Ticker']:
    df = pd.read_csv(f"D:\Data\Stock_SPOT\Day\TV_stocks data_cleaned\{ticker}")
    df['Date'] = pd.to_datetime(df['Datetime']).dt.date
    # if '_' in ticker:
    #     if 'M_M' in ticker:
    #         ticker = ticker.replace('_', '&')
        
    #     else:
    #         ticker = ticker.replace('_', '-')
    
    
        
    # if ticker == 'ATGL.csv':
    #     tdf = pd.read_csv(f"D:\Data\Stock_SPOT\Adjustment Completed\All\\ADANIGAS.csv")
    # else:
    #     tdf = pd.read_csv(f"D:\Data\Stock_SPOT\Adjustment Completed\All\\{ticker}")
    
    tdf = pd.read_csv(f"D:\Data\Stock_SPOT\Final_Adjustment2\{ticker}")
    
    # if 'Date/Time' in tdf.columns:
    #     tdf.rename(columns = {'Date/Time' : 'Datetime'}, inplace=True)
    # try:
    #     tdf['Datetime'] = pd.to_datetime(tdf['Datetime'], format = '%d-%m-%Y %H:%M:%S')
    # except:
    #     tdf['Datetime'] = pd.to_datetime(tdf['Datetime'], format = '%Y-%m-%d %H:%M:%S')
    
    tdf['Datetime'] = pd.to_datetime(tdf['Datetime'])
    
    tdf.set_index('Datetime', inplace=True)
    tdf = tdf.resample('1d').agg({'Open' : 'first', 'High' : 'max', 'Low' : 'min', 'Close' : 'last', 'Volume' : 'sum'}).dropna()
    
    
    tdf.reset_index(inplace=True)
    tdf['Date'] = tdf['Datetime'].dt.date

    
    
    ydf = yf.download(f"{ticker.replace('csv', 'NS')}", period= 'max')
    
    ydf.reset_index(inplace=True)
    
    ydf['Date'] = ydf['Date'].dt.date
    
    main_df = pd.merge(df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']], ydf[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']], on = ['Date'], suffixes=['_T', '_Y'], how = in_out)
    
    main_df = pd.merge(tdf[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']], main_df, on = ['Date'], how = in_out)
    
