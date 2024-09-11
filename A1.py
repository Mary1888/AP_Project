import numpy as np 
import pandas as pd 
import openpyxl

returns=pd.read_excel('C:/Users/75590/Desktop/BSC2/Master/Asset pricing/AP_Project/25_Portfolios_5x5_Wout_Div.xlsx', sheet_name='Avg Mon Value Weighted')

print(returns['Date'])