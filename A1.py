#%%
import numpy as np 
import pandas as pd 
import openpyxl
import matplotlib.pyplot as plt
#%%
data_returns=pd.read_excel('C:/Users/75590/Desktop/BSC2/Master/Asset pricing/AP_Project/25_Portfolios_5x5_Wout_Div.xlsx', sheet_name='Double Sort Jun', index_col=None)
data_canvas=pd.read_excel('C:/Users/75590/Desktop/BSC2/Master/Asset pricing/AP_Project/Data_Assignment_SMALLER.xlsx', sheet_name='FamaFrench Factors', index_col=None)

#196309-202110 
data_returns=data_returns[(data_returns['Date']<=202110) &  (data_returns['Date']>=196309)]
data_canvas.set_index('Date', inplace=True)
data_returns.set_index('Date',inplace=True)

market_port=data_canvas['Mkt-RF']
rf=data_canvas['RF']

# %%
#Calculate excess return for the 25 portfolios
adjusted_returns = data_returns.subtract(rf, axis=0)
# %%
adjusted_returns['Market']=market_port
# %%
#A.1 2 Mean, Variance and Correlation
mean_size_value=adjusted_returns.mean()
variance_size_value=adjusted_returns.var()
corr_size_value=adjusted_returns.corr()
# %%
# A.1 3 Mean-variance frontier without riskless assets
#mu
def Mu(returns,rf=0.15):
    return returns.mean()+rf
# lambda  
def Lambda(returns, mu_targ):
    mu=Mu(returns)
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
    mu=Mu(returns)
    return sigma_inv@mu/(mu.T@sigma_inv@vec_1)

def pi_mv(returns, mu_targ):
    lam=Lambda(returns, mu_targ)
    Pi_gmv=pi_gmv(returns)
    Pi_mu=pi_mu(returns)
    return lam*Pi_mu+(1-lam)*Pi_gmv
#adjust directly for lambda
def pi_mv_new(returns, lam):  
    Pi_gmv=pi_gmv(returns)
    Pi_mu=pi_mu(returns)
    return lam*Pi_mu+(1-lam)*Pi_gmv

# %%
values_targ=np.arange(-2,2,0.01)
mu_mv=list()
vol_mv=list()
for mu_targ in values_targ:
    mu=pi_mv(adjusted_returns, mu_targ).T@Mu(adjusted_returns)
    std=(pi_mv(adjusted_returns, mu_targ).T@adjusted_returns.cov()@pi_mv(adjusted_returns,mu_targ))**0.5
    mu_mv.append(mu)
    vol_mv.append(std)
    
# %%
#Plotting mean-variance frontier
plt.figure(figsize=(10, 6)) 
plt.plot(vol_mv, mu_mv, color='blue', linestyle='-')
plt.title('Mean-variance frontier without riskless assets')
plt.xlabel('volatility')
plt.ylabel('mu')
plt.legend()

#%%
#A.1 4 Mean-vairance frontier with riskless assets
#Tangency portfolio 
def pi_mv_riskless(returns,mu_targ):
    mu=Mu(returns,0)
    sigma_inv=np.linalg.inv(returns.cov())
    Pi_mu=mu_targ*sigma_inv@mu/(mu.T@sigma_inv@mu)
    return Pi_mu

def pi_tang(returns):
    mu=Mu(returns,0)
    sigma_inv=np.linalg.inv(returns.cov())
    vec_1=np.ones(len(mu))
    Pi_tang=sigma_inv@mu/(vec_1.T@sigma_inv@mu)
    return Pi_tang
# %%
values_targ=np.arange(-2,2,0.01)
mu_riskless_mv=list()
vol_riskless_mv=list()
for mu_targ in values_targ:
    mu=pi_mv_riskless(adjusted_returns, mu_targ).T@Mu(adjusted_returns)
    std=(pi_mv_riskless(adjusted_returns, mu_targ).T@adjusted_returns.cov()@pi_mv_riskless(adjusted_returns,mu_targ))**0.5
    mu_riskless_mv.append(mu)
    vol_riskless_mv.append(std)
    
# %%
#Plotting mean-variance frontier
plt.figure(figsize=(10, 6)) 
plt.plot(vol_mv, mu_mv, color='blue', linestyle='-')
plt.plot(vol_riskless_mv, mu_riskless_mv, color='red', linestyle='-')
plt.title('Mean-variance frontier with riskless assets')
plt.xlabel('volatility')
plt.ylabel('mu')
plt.legend()



