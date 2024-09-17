#%%
import numpy as np 
import pandas as pd 
import openpyxl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy.stats import f
from scipy import stats
#%%
######################################### A1 ################################################################
data_returns=pd.read_excel('25_Portfolios_5x5_Wout_Div.xlsx', sheet_name='Avg Mon Value Weighted', index_col=None)
data_canvas=pd.read_excel('Data_Assignment_SMALLER.xlsx', sheet_name='FamaFrench Factors', index_col=None)

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
values_targ=np.arange(-10,10,0.01)
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
values_targ=np.arange(-10,10,0.01)
mu_riskless_mv=list()
vol_riskless_mv=list()
for mu_targ in values_targ:
    mu=pi_mv_riskless(adjusted_returns, mu_targ).T@Mu(adjusted_returns)
    std=(pi_mv_riskless(adjusted_returns, mu_targ).T@adjusted_returns.cov()@pi_mv_riskless(adjusted_returns,mu_targ))**0.5
    mu_riskless_mv.append(mu)
    vol_riskless_mv.append(std)
   
mu_tang=pi_tang(adjusted_returns)@Mu(adjusted_returns)
vol_tang=(pi_tang(adjusted_returns).T@adjusted_returns.cov()@pi_tang(adjusted_returns))**0.5

# %%
#Plotting mean-variance frontier
plt.figure(figsize=(10, 6)) 
plt.plot(vol_mv, mu_mv, color='blue', linestyle='-')
plt.plot(vol_riskless_mv, mu_riskless_mv, color='red', linestyle='-')
plt.scatter(vol_tang,mu_tang, color='g',marker='o', s=100, label='Tangency portfolios')
plt.title('Mean-variance frontier with riskless assets')
plt.xlabel('volatility')
plt.ylabel('mu')
plt.legend()


#%%
#A.1 5 Regression 
alpha_hat=list()
residual_hat=np.zeros(data_returns.shape)
for i, var in enumerate(adjusted_returns.columns[:-1]):
    X=sm.add_constant(adjusted_returns[var])
    model=sm.OLS(adjusted_returns['Market'],X).fit()
    print('Coefficient: \n', model.params)
    print('Standard Error: \n', model.bse)
    print('Residuals: \n', model.resid)
    alpha_hat.append(model.params['const'])
    residual_hat[:,i]=model.resid

#%%
#A.1 6 GRS
T=data_returns.shape[0]
n=data_returns.shape[1]
sigma_hat=residual_hat.T@residual_hat/(T-2)
alpha_hat=np.array(alpha_hat)
X=adjusted_returns.drop(columns=['Market'])
X=np.column_stack((np.ones(len(X)), X))
q_11=np.linalg.inv(X.T@X)[0,0]

z=(T-n-1)*alpha_hat.T@np.linalg.inv(sigma_hat)@alpha_hat/n*(T-2)*q_11

p_value=1-f.cdf(z,n,T-n-1)
#reject H0, CAPM doesn't hold

# %%
####################################### A2 ##########################################
#A.2.1 Size portfolio Small-Big
size_data=pd.read_excel('Data_Assignment_SMALLER.xlsx', sheet_name='Size portfolios', index_col=None)
size_data.set_index('Date', inplace=True)
size_port=size_data['Lo 10']-size_data['Hi 10']-rf
mean_SMB=size_port.mean()
var_SMB=size_port.var()
#%%
#A.2.2 Regression 
adjusted_returns['SMB']=size_port
X=sm.add_constant(adjusted_returns['SMB'])
model=sm.OLS(adjusted_returns['Market'],X).fit()
print('Coefficient: \n', model.params)
print('Standard Error: \n', model.bse)
print('Residuals: \n', model.resid)

#%%
#A.2.3 GRS
alpha_hat=list()
residual_hat=np.zeros(data_returns.shape)
for i, var in enumerate(adjusted_returns.columns[:-2]):
    X=sm.add_constant(adjusted_returns[var])
    model=sm.OLS(adjusted_returns['SMB'],X).fit()
    print('Coefficient: \n', model.params)
    print('Standard Error: \n', model.bse)
    print('Residuals: \n', model.resid)
    alpha_hat.append(model.params['const'])
    residual_hat[:,i]=model.resid

T=data_returns.shape[0]
n=data_returns.shape[1]
sigma_hat=residual_hat.T@residual_hat/(T-2)
alpha_hat=np.array(alpha_hat)
X=adjusted_returns.drop(columns=['Market','SMB'])
X=np.column_stack((np.ones(len(X)), X))
q_11=np.linalg.inv(X.T@X)[0,0]
z=(T-n-1)*alpha_hat.T@np.linalg.inv(sigma_hat)@alpha_hat/n*(T-2)*q_11
p_value=1-f.cdf(z,n,T-n-1)  
    
    
#%% 
##################################### A3 ###########################
#%%
#A.3.1 Momentum t-12 to t-2
cumulative_returns = ((data_returns/100).shift(2) + 1).rolling(window=11).apply(lambda x: x.prod() - 1, raw=True)
cumulative_returns=cumulative_returns*100


#%%
#A.3.2 Fama-MacBeth Step 1
port_returns=adjusted_returns.drop(columns=['Market','SMB'])
slope=[]
for i in range(12,len(cumulative_returns)):
    X=cumulative_returns.iloc[i].values
    y=port_returns.iloc[i].values
    X=sm.add_constant(X)
    model=sm.OLS(y,X).fit()
    slope.append(float(model.params[1:]))

# %%
slope_df=cumulative_returns[12:]
slope_df['Slope']=slope
#Plot of the slope over time
plt.figure(figsize=(10, 6)) 
plt.plot(pd.to_datetime(slope_df.index, format='%Y%m'),slope_df['Slope'], color='blue', linestyle='-')
plt.title('Slope coefficients from the cross-sectional regression')
plt.xlabel('Time')
plt.ylabel('Slope coefficient')
plt.legend()

# %%
#A.3.3 Fama-MacBeth Step 2 
mean_slope=slope_df['Slope'].mean()
t_statistic, p_value = stats.ttest_1samp(slope_df['Slope'], 0)
