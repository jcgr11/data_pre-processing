import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import numpy as np

#Created Correlation Matrix and heatmap, see corr_matrx.py file
#list of ETFs
tickers = ['VTI','GLD','BND','VT','IYR','AIA','VEU','AAXJ','ILF','EZU','EEM','FM','VWO','UUP','GSG', 'BWX']
len(tickers)
#Choose Dates
START_DATE = '2013-01-24'
END_DATE = '2023-01-24'
#monthly returns
df = yf.download(tickers, start=START_DATE, end=END_DATE)['Adj Close']\
 .resample('M') \
 .last() \
 .pct_change() \
 .dropna()

#create correlation matrix and iterate over matrix to print final output for top correlation pairs in rank order. function called at end of script
def corrank(X: df):
        import itertools
        df = pd.DataFrame([[(i,j),X.corr().loc[i,j]] for i,j in list(itertools.combinations(X.corr(), 2))],columns=['pairs','corr'])    
        print(df.sort_values(by='corr',ascending=False).head(20)[['pairs', 'corr']])

#setup heat map design
cmap = sns.diverging_palette(220, 150, as_cmap=True)

#mask redundancy
mask = np.triu(np.ones_like(df.corr(), dtype=bool))
np.fill_diagonal(mask, False)

#refine heatmap
sns.heatmap(
        data=df.corr(), linewidths=0.3,  #width of lines separating the matrix cells
        square=True,cmap=cmap,
        vmax=1,  #define the max of corr scale
        vmin=-1, #define the min of corr scale
        center=0, cbar_kws={"shrink": .75}, mask=mask, annot=True)

plt.yticks(rotation=0) #adjust y-axis ticks 

print("Top 20 Absolute Correlations")
corrank(df) #top correlation pairs
plt.show() #final heat map


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
