import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#list of ETFs
tickers = ['VTI','GLD','BND','VT','IYR','AIA','VEU','AAXJ','ILF','EZU','EEM','FM','VWO','UUP','GSG', 'BWX']
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
cmap = sns.diverging_palette(230, 20, as_cmap=True)

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
