import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

ticker = 'AXP'
START_DATE = '2020-03-20' #22nd fell on a sunday, price used was for friday the 20th
END_DATE = '2023-03-23'

x = yf.download(ticker, start=START_DATE, end=END_DATE)['Close']\
 .resample('D') \
 .last() \
 .pct_change() \
 .dropna()

x1 = x.reset_index() # move date and time to axis 1 index 0
date = x1['Date'].dt.date # remove time stamp
ex_dt = x1.iloc[:,1:] # create new date only index column
data = ex_dt.set_index(date) # set date column
data
# Geometric linking of daily returns
geometric_means = []
for col in data:
    geometric_mean = ((data[col]+1)).prod()-1
    geometric_means.append(geometric_mean)

total = pd.DataFrame([geometric_means], columns=data.columns, index=['3 yr Geometric Ret.(Close Price)'])
total.rename(columns={'Close': ''}, inplace=True)
total
x2=data.loc[~(data==0).all(axis=1)] #drop zero values from non-trading days

x2.rename(columns={'Close': 'Daily Return'}, inplace=True) #reset column for plot output

x2.plot(title=f'{ticker} Daily returns: {START_DATE} - {END_DATE}') #plot title
plt.figtext(.825, .82, f'3 yr Return:{100 * total.iloc[0,0]:.2f}%')
print(total)
plt.show()