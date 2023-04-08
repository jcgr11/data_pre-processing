import pandas as pd
import yfinance as yf
import statsmodels.api as sm

RISKY_ASSET = 'VTI'
MARKET_BENCHMARK = 'SPY'
MKT_RF ='SHY'
START_DATE = '2013-01-24'
END_DATE = '2023-01-24'

# create data frame of timeseries for asset, benchmark, and risk free rate proxy
df = yf.download([RISKY_ASSET, MARKET_BENCHMARK, MKT_RF],start=START_DATE,end=END_DATE,progress=False)
# calculate returns
X = df['Adj Close'].rename(columns={RISKY_ASSET: 'asset', MARKET_BENCHMARK: 'market', MKT_RF: 'mkt_rf'}) \
 .resample('M') \
 .last() \
 .pct_change() \
 .dropna()


rf_series = X.loc[:,'mkt_rf'] # SHY return series 
X1 = X.subtract(rf_series, axis=0) # subtract RF return series from asset and benchmark series
X2= X1.drop(['mkt_rf'], axis=1) # Drop RF series (now all 0 values)

# covariance method for beta calculation (Cov(asset, benchmark) divided by volatility of market returns)
covariance = X2.cov().iloc[0,1]
benchmark_variance = X2.market.var()
beta = covariance / benchmark_variance
beta

# CAPM regression setup
y = X2.pop('asset') #extract dependent variable (asset) times series
X2 = sm.add_constant(X2) # independent variable (benchmark) time series
capm_model = sm.OLS(y, X2).fit() # OLS regression
capm_model.summary() # Regression output. market coefficient should equal beta variable above from covariance method



