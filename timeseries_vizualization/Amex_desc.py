import pandas as pd
import numpy as np
import yfinance as yf
import scipy.stats as scs

df = yf.download('AXP', start='2020-03-20',end='2023-03-23', progress=False)
df = df.loc[:, ['Adj Close']]
df.rename(columns={'Adj Close':'adj_close'}, inplace=True)

df['simple_rtn'] = df.adj_close.pct_change()
df['log_rtn'] = np.log(df.adj_close/df.adj_close.shift(1))

r_range = np.linspace(min(df.log_rtn), num=1000)
mu = df.log_rtn.mean()
sigma = df.log_rtn.std()
norm_pdf = scs.norm.pdf(r_range, loc = mu, scale = sigma)

df.log_rtn.describe()
