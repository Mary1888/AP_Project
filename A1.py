#%%
import numpy as np 
import pandas as pd 
import openpyxl
import matplotlib.pyplot as plt
#%%

data_returns=pd.read_excel('C:/Users/75590/Desktop/BSC2/Master/Asset pricing/AP_Project/25_Portfolios_5x5_Wout_Div.xlsx', sheet_name='Double Sort Jun', index_col=None)
data_canvas=pd.read_excel('C:/Users/75590/Desktop/BSC2/Master/Asset pricing/AP_Project/Data_Assignment_SMALLER.xlsx', sheet_name='FamaFrench Factors', index_col=None)
data_returns=data_returns[(data_returns['Date']<=202110) &  (data_returns['Date']>=196309)]

data_canvas.set_index('Date', inplace=True)
data_returns.set_index('Date',inplace=True)

market_port=data_canvas['Mkt-RF']
rf=data_canvas['RF']

#196309-202110 
# %%
adjusted_returns = data_returns.subtract(rf, axis=0)
# %%
adjusted_returns['Market']=market_port
# %%
#A.1 2 Mean, Variance and Correlation
mean_size_value=adjusted_returns.mean()
variance_size_value=adjusted_returns.var()
corr_size_value=adjusted_returns.corr()
# %%
# A.1 3 Mean-variance frontier

#pi_gmv 
sigma=adjusted_returns.cov()
vec_1=np.ones(26)
sigma_inv=np.linalg.inv(sigma)
pi_gmv=sigma_inv@vec_1/(vec_1.T@sigma_inv@vec_1)
#pi_mu
mu=mean_size_value
pi_mu=sigma_inv@mu/(mu.T@sigma_inv@vec_1)



#%%
# lambda  
def Lambda(returns, mu_targ):
    mu=returns.mean()+0.15
    sigma=returns.cov()
    sigma_inv=np.linalg.inv(sigma)
    vec_1=np.ones(len(mu))
    B=mu.T@sigma_inv@vec_1
    C=vec_1.T@sigma_inv@vec_1
    A=mu.T@sigma_inv@mu
    l=(B*C*mu_targ-B**2)/(A*C-B**2)
    return l
#pi_gmv
def pi_gmv(returns):
    sigma=returns.cov()
    vec_1=np.ones(26)
    sigma_inv=np.linalg.inv(sigma)
    return sigma_inv@vec_1/(vec_1.T@sigma_inv@vec_1)
#pi_mu
def pi_mu(returns):
    sigma=returns.cov()
    vec_1=np.ones(26)
    sigma_inv=np.linalg.inv(sigma)
    mu=returns.mean()+0.15
    return sigma_inv@mu/(mu.T@sigma_inv@vec_1)

def pi_mv(returns, mu_targ):
    lam=Lambda(returns, mu_targ)
    Pi_gmv=pi_gmv(returns)
    Pi_mu=pi_mu(returns)
    return lam*Pi_mu+(1-lam)*Pi_gmv

def pi_mv_new(returns, lam):
    Pi_gmv=pi_gmv(returns)
    Pi_mu=pi_mu(returns)
    return lam*Pi_mu+(1-lam)*Pi_gmv

#%%
pi_mv(adjusted_returns, 0.1)

#%%
Lambda(adjusted_returns,0.5)
# %%
values_targ=np.arange(-2,2,0.001)
mu_mv=list()
var_mv=list()
for mu_targ in values_targ:
    mu=pi_mv(adjusted_returns, mu_targ).mean()
    variance=pi_mv(adjusted_returns, mu_targ).var()
    mu_mv.append(mu)
    var_mv.append(variance)
    
# %%
plt.figure(figsize=(10, 6))  # Optional: Set the figure size

# Plot the first series
plt.plot(var_mv, mu_mv, color='blue', linestyle='-')

# Add title and labels
plt.title('Plot')
plt.xlabel('volatility')
plt.ylabel('mu')


# Add a legend
plt.legend()
# %%
adjusted_returns.mean()
# %%
