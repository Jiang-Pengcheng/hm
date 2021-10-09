# -*- codeing = utf-8 -*-
# @Time :2021/9/30 2:15
# @Author:JPC
# @File :555.py
# @software: PyCharm
import pandas as pd
import numpy as np
from scipy import  stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
import itertools

import itertools
from statsmodels.tsa.arima_model import ARIMA
df=pd.read_csv('宏观.csv')
dfx=pd.read_csv('211001.csv')

Y=dfx.iloc[0:9,-4]
Y_train = Y

p=q=range(0,2)
d = range(0,2)
pdq=list(itertools.product(p,d,q))
seasonal_pdq=[(x[0],x[1],x[2],12) for x in pdq]


#根据AIC值确定最佳参数，AIC越小越好
for param in pdq:
    for param_seasonal in seasonal_pdq:
        mod = sm.tsa.statespace.SARIMAX(Y_train,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
        results = mod.fit()
        print('ARIMA{}x{}1 - AIC:{}'.format(param, param_seasonal, results.aic))

#根据上述确定参数建模
Y_train = Y.tolist()
param = (1,1,1)
param_seasonal = (1, 0, 1, 12)
mod = sm.tsa.statespace.SARIMAX(Y_train,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
results = mod.fit()
Y_predict = results.forecast(9)
print(Y_predict)
for item in Y_predict:
    Y_train.append(item)

Y_train=pd.DataFrame(Y_train)
Y_train.to_csv('ss.csv')

