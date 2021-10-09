# -*- codeing = utf-8 -*-
# @Time :2021/9/30 2:04
# @Author:JPC
# @File :yuce.py
# @software: PyCharm
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from scipy import  stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
import itertools
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression,Ridge, RidgeCV


df=pd.read_csv('原始正确.csv')
df1=pd.read_csv('技术-.csv')


scaler = MinMaxScaler(feature_range=[5,10])
result = scaler.fit_transform(df)
result=pd.DataFrame(result)


x=result.iloc[:9,3:]
y=df.iloc[:9,2]

model = Ridge(1)

model.fit(x,y)
# print('系数矩阵:\n',model.coef_)



x_forecast=df1.iloc[:,1:]
yp=model.predict(x_forecast)
yp=pd.DataFrame(yp)



yp.to_csv('yp.csv')


