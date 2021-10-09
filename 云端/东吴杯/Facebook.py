# -*- codeing = utf-8 -*-
# @Time :2021/9/29 19:59
# @Author:JPC
# @File :Facebook.py
# @software: PyCharm
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import tqdm
import copy
df=pd.read_csv('宏观.csv')
dfx=pd.read_excel('特征.xlsx')

dfxx=copy.deepcopy(dfx)
for i in range(11):

    for j in range(9):

        dft = dfxx.iloc[ 0:j+9, i ]
        # 一次指数平滑预测
        def es1(list3, t, a):
            if t == 0:
                return list3[ 0 ]  # 初始的平滑值取实际值

            return a * list3[ t - 1 ] + (1 - a) * es1(list3, t - 1, a)  # 递归调用 t-1 → 12



        a = 0.8  # 平滑常数
        t = len(dft)  # 预测的时期 13
        x = es1(dft, t, a)



        dfxx.loc[j+9,i]=x







