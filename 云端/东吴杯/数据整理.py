# -*- codeing = utf-8 -*-
# @Time :2021/9/24 15:01
# @Author:JPC
# @File :数据整理.py
# @software: PyCharm
import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

import copy
dataset =pd.read_excel('temp.xlsx')


def fill_miss_byRandomForest(data_df , obj_column, missing_other_column ):
    ## 先把有缺失的其他列删除掉missing_other_column
    data_df = data_df.drop(missing_other_column , axis = 1)
    # 分成已知该特征和未知该特征两部分
    known = data_df[data_df[obj_column].notnull()]
    unknown = data_df[data_df[obj_column].isnull()]
    # y为结果标签值
    y_know = known[obj_column]
    # X为特征属性值
    X_know= known.drop(obj_column , axis = 1)
    from sklearn.ensemble import RandomForestRegressor
    rfr = RandomForestRegressor(random_state=0, n_estimators=70,max_depth=3,n_jobs=-1)
    rfr.fit(X_know,y_know)
    # 用得到的模型进行未知特征值预测
    # X为特征属性值
    X_unknow= unknown.drop(obj_column , axis = 1)
    predicted = rfr.predict(X_unknow).round(0)
    data_df.loc[(data_df[obj_column].isnull()), obj_column] = predicted
    return data_df


num_na_back = pd.isna(dataset).sum()
obj_column = pd.DataFrame(num_na_back)
obj_column.drop(obj_column[ obj_column[ 0 ] == 0 ].index, inplace = True, axis = 0)
len_obj_column = obj_column.index.to_list()


len_obj_column_copy = copy.deepcopy(len_obj_column)

dataset_copy = dataset
for j in len_obj_column:
    len_obj_column_copy.remove(j)
    Balance_row_temp_1 = fill_miss_byRandomForest(dataset_copy, j, len_obj_column_copy)
    Balance_row_temp_1 = Balance_row_temp_1[ j ]
    dataset_copy = dataset_copy.drop(j, axis = 1)
    dataset_copy = dataset_copy.merge(Balance_row_temp_1, how = 'inner', left_index = True, right_index = True)

dataset_copy.to_excel('temp67.xlsx')