#%%
import numpy as np 
import pandas as pd 
import openpyxl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy.stats import f
from scipy import stats
import seaborn
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
#adjusted_returns.applymap(lambda x: (x / 100) + 1) #adjust for gross return in case necessary
# %%
#A.1 2 Mean, Variance and Correlation

mean_size_value=adjusted_returns.mean()
variance_size_value=adjusted_returns.var()
corr_size_value=adjusted_returns.corr()
#heatmap for correlation 
plt.figure(figsize=(8, 6))
seaborn.heatmap(corr_size_value, annot=False, cmap='YlGnBu')
plt.show()
# %%
# A.1 3 Mean-variance frontier without riskless assets
#mu
def Mu(returns,rf=0.0015):
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
pi_mu(adjusted_returns)
#%%
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
    X=sm.add_constant(adjusted_returns['Market'])
    model=sm.OLS(adjusted_returns[var],X).fit()
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
X=np.column_stack((np.ones(len(X)), adjusted_returns['Market']))
q_11=np.linalg.inv(X.T@X)[0,0]

z=(T-n-1)*alpha_hat.T@np.linalg.inv(sigma_hat)@alpha_hat/n*(T-2)*q_11

p_value=1-f.cdf(z,n,T-n-1)
#reject H0, CAPM doesn't hold

# %%
####################################### A2 ##########################################
#A.2.1 Size portfolio Small-Big
size_data=pd.read_excel('Data_Assignment_SMALLER.xlsx', sheet_name='Size portfolios', index_col=None)
size_data.set_index('Date', inplace=True)
size_port=size_data['Lo 10']-size_data['Hi 10']
mean_SMB=size_port.mean()
var_SMB=size_port.var()
#%%
#A.2.2 Regression 
adjusted_returns['SMB']=size_port
X=sm.add_constant(adjusted_returns['Market'])
model=sm.OLS(adjusted_returns['SMB'],X).fit()
print('Coefficient: \n', model.params)
print('Standard Error: \n', model.bse)
print('Residuals: \n', model.resid)

#%%
#A.2.3 GRS
alpha_hat=list()
residual_hat=np.zeros(data_returns.shape)
for i, var in enumerate(adjusted_returns.columns[:-2]):
    X=sm.add_constant(adjusted_returns[['Market','SMB']])
    model=sm.OLS(adjusted_returns[var],X).fit()
    print('Coefficient: \n', model.params)
    print('Standard Error: \n', model.bse)
    print('Residuals: \n', model.resid)
    alpha_hat.append(model.params['const'])
    residual_hat[:,i]=model.resid
k=2
T=data_returns.shape[0]
n=data_returns.shape[1]
sigma_hat=residual_hat.T@residual_hat/T
alpha_hat=np.array(alpha_hat)
factors=adjusted_returns[['Market','SMB']]
mu_fa=factors.mean()
var_fa=factors.cov()
z=(T-n-k)*alpha_hat.T@np.linalg.inv(sigma_hat)@alpha_hat/(n*(1+mu_fa.T@np.linalg.inv(var_fa)@mu_fa))
p_value=1-f.cdf(z,n,T-n-k)  
    

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


#%%
############################ A4 ################################
#%%
#A.4.1 PCA scree plot
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

data=data_returns.subtract(rf, axis=0)
X_train=data[:int(len(data)/2)]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
# Apply PCA
pca = PCA()
pca.fit(X_scaled)

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)
#Adjust for percentage of variance retained 
confidence_level=0.92
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.axhline(y=confidence_level, color='r', linestyle='--')  # 90% threshold line
plt.show()

n_components = np.argmax(cumulative_variance >= confidence_level) + 1
print(f'Number of components to retain 92% variance: {n_components}')


#%%
#A.4.2 PCA 3-factor model
pca = PCA(n_components=3)
principal_components = pca.fit_transform(X_scaled)
loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3'])
loadings.index = X_train.columns  
print(loadings)
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

sns.barplot(ax=axes[0], x=loadings.index, y=loadings['PC1'])
axes[0].set_title('PC1 Loadings')
axes[0].set_xticklabels(loadings.index, rotation=90)

sns.barplot(ax=axes[1], x=loadings.index, y=loadings['PC2'])
axes[1].set_title('PC2 Loadings')
axes[1].set_xticklabels(loadings.index, rotation=90)

sns.barplot(ax=axes[2], x=loadings.index, y=loadings['PC3'])
axes[2].set_title('PC3 Loadings')
axes[2].set_xticklabels(loadings.index, rotation=90)
plt.suptitle('Factor Loadings of Each Asset on the First 3 Principal Components')
plt.tight_layout()
plt.show()

pca_df = pd.DataFrame(data=principal_components, columns= [f'PC{i+1}' for i in range(n_components)])
print(pca_df)

#%%
#A.4.3 Fama French 3-factor model 
fama_data=pd.read_excel('Data_Assignment_SMALLER.xlsx', sheet_name='FamaFrench Factors', index_col=None)
fama_data.set_index('Date', inplace=True)
fama_data=fama_data.drop(columns=['RF'])
fama_train=fama_data[:int(len(fama_data)/2)]
mean_fama=fama_train.mean()
var_fama=fama_train.var()
corr_fama=fama_train.corr()
sp_fama=mean_fama/(var_fama**0.5)
#%%
#Statistical properties PCA factors
mean_pca=pca_df.mean()
var_fama=pca_df.var()
corr_fama=pca_df.corr()
sp_pca=mean_pca/(var_fama**0.5)

#%% 
#A.4.4 GRS 
#GRS on PCA factors 
X_test=data[int(len(data)/2):]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_test)
pca = PCA(n_components=3)
pc_test = pca.fit_transform(X_scaled)
pca_test_df = pd.DataFrame(data=pc_test, columns= [f'PC{i+1}' for i in range(n_components)])
pca_test_df.index=X_test.index
alpha_hat=list()
residual_hat=np.zeros(X_test.shape)
for i, var in enumerate(X_test.columns):
    X=sm.add_constant(pca_test_df[['PC1','PC2','PC3']])
    model=sm.OLS(X_test[var],X).fit()
    print('Coefficient: \n', model.params)
    print('Standard Error: \n', model.bse)
    print('Residuals: \n', model.resid)
    alpha_hat.append(model.params['const'])
    residual_hat[:,i]=model.resid
k=3
T=X_test.shape[0]
n=X_test.shape[1]
sigma_hat=residual_hat.T@residual_hat/T
alpha_hat=np.array(alpha_hat)
factors=pca_test_df
mu_fa=factors.mean()
var_fa=factors.cov()
z=(T-n-k)*alpha_hat.T@np.linalg.inv(sigma_hat)@alpha_hat/(n*(1+mu_fa.T@np.linalg.inv(var_fa)@mu_fa))
p_value=1-f.cdf(z,n,T-n-k)  

#%%
#GRS on FamaFrench 3 factors 
X_test=data[int(len(data)/2):]
fama_test=fama_data[int(len(fama_data)/2):]
alpha_hat=list()
residual_hat=np.zeros(X_test.shape)
for i, var in enumerate(X_test.columns):
    X=sm.add_constant(fama_test)
    model=sm.OLS(X_test[var],X).fit()
    print('Coefficient: \n', model.params)
    print('Standard Error: \n', model.bse)
    print('Residuals: \n', model.resid)
    alpha_hat.append(model.params['const'])
    residual_hat[:,i]=model.resid
k=3
T=X_test.shape[0]
n=X_test.shape[1]
sigma_hat=residual_hat.T@residual_hat/T
alpha_hat=np.array(alpha_hat)
factors=pca_test_df
mu_fa=factors.mean()
var_fa=factors.cov()
z=(T-n-k)*alpha_hat.T@np.linalg.inv(sigma_hat)@alpha_hat/(n*(1+mu_fa.T@np.linalg.inv(var_fa)@mu_fa))
p_value=1-f.cdf(z,n,T-n-k) 

################################## A5 #####################
#%%
#A.5.1 Realized volatility regression 
rv_data=pd.read_excel('Data_Assignment_SMALLER.xlsx', sheet_name='Realized Volatility', index_col=None)
rv_data.set_index('Date', inplace=True)

adjusted_returns = data_returns.subtract(rf, axis=0)
adjusted_returns['Market']=market_port
adjusted_returns['RV']=rv_data

#Time-series regression 
beta_hat=list()
residual_hat=np.zeros(data_returns.shape)
for i, var in enumerate(adjusted_returns.columns[:-2]):
    X=sm.add_constant(adjusted_returns[['Market','RV']])
    model=sm.OLS(adjusted_returns[var],X).fit()
    print('Coefficient: \n', model.params)
    print('Standard Error: \n', model.bse)
    print('Residuals: \n', model.resid)
    beta_hat.append([model.params['Market'],model.params['RV']])
    residual_hat[:,i]=model.resid
beta_hat_df=pd.DataFrame(beta_hat, columns=['Market','RV'])
beta_hat_df.index=adjusted_returns.columns[:-2]
# %%
#A.5.2 
average_excess_returns=adjusted_returns.drop(columns=['Market', 'RV']).mean()
avg_and_beta_df=beta_hat_df
avg_and_beta_df['Average excess return']=average_excess_returns

#%%
#OLS
X=avg_and_beta_df[['Market','RV']]
model=sm.OLS(avg_and_beta_df['Average excess return'],X).fit()
print('Coefficient: \n', model.params)
print('Standard Error: \n', model.bse)
print('Residuals: \n', model.resid)
lambda_ols=np.array(model.params)
#%%
#GLS 
sigma_hat=residual_hat.T@residual_hat/len(residual_hat) #from time-series regression
model=sm.GLS(avg_and_beta_df['Average excess return'],X, sigma=sigma_hat).fit()
print('Coefficient: \n', model.params)
print('Standard Error: \n', model.bse)
print('Residuals: \n', model.resid)
alpha_hat_GLS=np.array(model.resid)
lambda_gls=np.array(model.params)
# %%
#Testing OLS
T=len(residual_hat)
k=2
sigma_f=beta_hat_df[['Market','RV']].cov()
J_ols=T*(1/(1+lambda_ols.T@np.linalg.inv(sigma_f)@lambda_ols))*alpha_hat_GLS.T@np.linalg.inv(sigma_hat)@alpha_hat_GLS
p_value = stats.chi2.sf(J_ols,T-k)

#%%
#Testing GLS
J_gls=T*(1/(1+lambda_gls.T@np.linalg.inv(sigma_f)@lambda_gls))*alpha_hat_GLS.T@np.linalg.inv(sigma_hat)@alpha_hat_GLS
p_value = stats.chi2.sf(J_gls,T-k)

# %%
