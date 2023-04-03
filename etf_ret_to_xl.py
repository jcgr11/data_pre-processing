import yfinance as yf
import pandas as pd
import numpy as np
import os

# ETFs and date range
tickers = ['VNQ','BWX','ANGL','DBC','EEM','FDRV','REZ','FXE','EMB','USO','EEM','VT','TSLA','VTI']
START_DATE = '2022-02-28'
END_DATE = '2023-03-02'

# download historical adj close price and calculate returns
x = yf.download(tickers, start=START_DATE, end=END_DATE)['Adj Close']\
 .resample('D') \
 .last() \
 .pct_change() \
 .dropna()

# yahoo finance includes time in date-time for time series, must reset index to remove time for excel output
x1 = x.reset_index() # move date and time to axis 1 index 0
date = x1['Date'].dt.date # remove time stamp
ex_dt = x1.iloc[:,1:] # create new date only index column
excel_output = ex_dt.set_index(date) # set date column

# Geometric linking of daily returns
geometric_means = []
for col in excel_output:
    geometric_mean = ((excel_output[col]+1)).prod()-1
    geometric_means.append(geometric_mean)
total_row = pd.DataFrame([geometric_means], columns=excel_output.columns, index=['total'])
excel_output = pd.concat([excel_output, total_row])

excel_output.to_excel("HPR_ret.xlsx") # save and exports return streams to excel

# open returns file
returns = "C:\py_scripts/HPR_ret.xlsx"
os.startfile(returns)