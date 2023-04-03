import yfinance as yf 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

N_PORTFOLIOS = 10**5
N_DAYS = 252
    
RISKY_ASSETS = ['VTI','GLD','BND','VT','IYR','AIA','VEU','AAXJ','ILF','EZU','EEM','FM','VWO','UUP','GSG', 'BWX']
RISKY_ASSETS.sort() 

START_DATE = '2013-01-01' 
END_DATE = '2023-01-01' 

n_assets = len(RISKY_ASSETS)
prices_df = yf.download(RISKY_ASSETS, start = START_DATE, end = END_DATE) 
returns_df = prices_df['Adj Close'].pct_change().dropna() 
avg_returns = returns_df.mean() * N_DAYS
rf = 0.02

cov_mat = (returns_df - rf).cov() * N_DAYS

np.random.seed(10**5)
weights = np.random.uniform(-1, 1, size = (N_PORTFOLIOS, n_assets))
weights_abs = np.abs(weights)
weights_abs /= np.sum(weights_abs, axis = 1)[:, np.newaxis] # Ensure they add up to 1

## Ensuring all portfolio returns to greater than 0

# by setting negative weights to -100 and positive weights to 100
positive_weights = np.where(weights >= 0, 100, 0) 
negative_weights = np.where(weights < 0, -100, 0) 

# Adding negative_weights in portf_rtns calculation 
# which add potential for negative Sharpe ratio
portf_rtns = np.sum(weights * avg_returns, axis=1)

portf_vol = []
for i in range(0,len(weights)):
    portf_vol.append(np.sqrt(np.dot(weights[i].T, np.dot(cov_mat,weights[i]))))
    
portf_vol = np.array(portf_vol)
# Computing the Sharpe Ratio
portf_sharpe_ratio = (portf_rtns - (rf * N_DAYS)) / portf_vol
portf_sharpe_ratio = np.where(portf_sharpe_ratio<0, 0, portf_sharpe_ratio)

# Replacing negative Sharpe ratio to be equivalent to 0
portf_results_df = pd.DataFrame({'returns' : portf_rtns, 'volatility' : portf_vol, 'sharpe_ratio' : np.where(portf_sharpe_ratio<0, 0, portf_sharpe_ratio)})

# Computing and plotting the efficient frontier
N_POINTS = 100
portf_vol_ef = []
indices_to_skip = []
# Replacing returns to be double the negative values
portf_rtns_ef = np.linspace(portf_results_df.returns.min(), 
                            portf_results_df.returns.max()*2,N_POINTS)
portf_rtns_ef = np.round(portf_rtns_ef, 2)

# Replacing return values of negative Sharpe ratio values 
# to be double the negative values
portf_rtns = np.round(np.where(portf_rtns>0, 
                               portf_rtns, portf_rtns*2),2)

for point_index in range(N_POINTS):
    if portf_rtns_ef[point_index] not in portf_rtns:
        indices_to_skip.append(point_index)
        continue
        
    matched_ind = np.where(portf_rtns == portf_rtns_ef[point_index])
    portf_vol_ef.append(np.min(portf_vol[matched_ind]))
    
portf_rtns_ef = np.delete(portf_rtns_ef, indices_to_skip)

# Computing and plotting the Maximum Sharpe Ratio portfolio
# and Minimum Volatility portfolio
MARKS = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', 'P', '*', 'h']


fig, ax = plt.subplots()
portf_results_df.plot(kind = 'scatter', x = 'volatility', y = 'returns', c = 'sharpe_ratio', cmap = 'RdYlGn', edgecolors = 'black', ax = ax) 
ax.set(xlabel = 'Volatility', ylabel = 'Expected Returns', title = 'Efficient Frontier')
ax.plot(portf_vol_ef, portf_rtns_ef, 'b--') 

for asset_index in range(n_assets):
 ax.scatter(x=np.sqrt(cov_mat.iloc[asset_index, asset_index]), 
            y=avg_returns[asset_index], marker=MARKS[asset_index], s=150, color='red', label=RISKY_ASSETS[asset_index])
ax.legend()

max_sharpe_ind = np.argmax(portf_results_df.sharpe_ratio)
max_sharpe_portf = portf_results_df.loc[max_sharpe_ind]
min_vol_ind = np.argmin(portf_results_df.volatility)
min_vol_portf = portf_results_df.loc[min_vol_ind]

ax.scatter(x=max_sharpe_portf.volatility, 
            y=max_sharpe_portf.returns, c='black', marker='*', s=200, label='Max Sharpe Ratio')
ax.scatter(x=min_vol_portf.volatility, 
            y=min_vol_portf.returns, c='black', marker='P', s=200, label='Minimum Volatility')

# Print results
print('Maximum Sharpe ratio portfolio ----')
print('Performance')
for index, value in max_sharpe_portf.items():
 print(f'{index}: {100 * value:.2f}% ', end="", flush=True)
print('\nWeights')
for x, y in zip(RISKY_ASSETS, weights[np.argmax(portf_results_df.sharpe_ratio)]):
 print(f'{x}: {100*y:.2f}% ', end="", flush=True)

print('Minimum vol. portfolio ----')
print('Performance')
for index, value in min_vol_portf.items():
 print(f'{index}: {100 * value:.2f}% ', end="", flush=True)
print('\nWeights')
for x, y in zip(RISKY_ASSETS, weights[np.argmin(portf_results_df.volatility)]):
 print(f'{x}: {100*y:.2f}% ', end="", flush=True)

fig, ax = plt.subplots()
portf_results_df.plot(kind='scatter', x='volatility', y='returns', c='sharpe_ratio', cmap='RdYlGn', edgecolors='black', ax=ax)
ax.scatter(x=max_sharpe_portf.volatility, y=max_sharpe_portf.returns, c='black', marker='*', s=200, label='Max Sharpe Ratio')
ax.scatter(x=min_vol_portf.volatility, y=min_vol_portf.returns, c='black', marker='P', s=200, label='Minimum Volatility')
ax.set(xlabel='Volatility', ylabel='Expected Returns', title='Efficient Frontier')
ax.legend()
plt.show()