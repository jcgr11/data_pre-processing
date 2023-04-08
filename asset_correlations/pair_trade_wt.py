import yfinance as yf
import pandas as pd
import statsmodels.api as sm
import numpy as np

#Created Correlation Matrix and heatmap, see corr_matrx.py file
import corr_mtrx

pair = ['VT', 'EEM'] #choose pair from terminal output (I chose lower correlation)
#choose dates
START_DATE = '2013-01-31'
END_DATE = '2023-12-31'

#pull data and calculate returns
x = yf.download(pair, start=START_DATE, end=END_DATE)['Adj Close']\
 .resample('M') \
 .last() \
 .pct_change() \
 .dropna()

y = x.pop('EEM') #y is long position. you must type this in.
x = sm.add_constant(x) #short position
model = sm.OLS(y, x).fit() #OLS regression
model.summary() #regression output
coef = pd.read_html(model.summary().tables[1].as_html(),header=0,index_col=0)[0] #read regression table to pd df
m = coef['coef'].values[1] #get est. slope from regression df

wt_y = 1/(1-m) #optimal weight for long position
wt_x= 1-wt_y #optimal weight for short position

print(f'Estimated optimal weights for pair-trade: {wt_y} long position and {wt_x} short position')
