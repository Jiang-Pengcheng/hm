# -*- codeing = utf-8 -*-
# @Time :2021/9/10 9:02
# @Author:JPC
# @File :识别企业税务合规存疑.py
# @software: PyCharm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import seaborn as sns
from scipy.stats import chi
import scipy
from scipy.stats import chi2
np.seterr(divide='ignore',invalid='ignore')
df=pd.read_csv('tax.csv')
#标签label二分
df['label']=df['label'].apply(lambda x:0 if x=='正常' else 1)
#分箱


def chi3(arr):
    '''
   计算卡方值
   arr:频数统计表,二维numpy数组。
   '''
    assert (arr.ndim == 2)
    # 计算每行总频数
    R_N = arr.sum(axis = 1)
    # 每列总频数
    C_N = arr.sum(axis = 0)
    # 总频数
    N = arr.sum()
    # 计算期望频数 C_i * R_j / N。
    E = np.ones(arr.shape) * C_N / N
    E = (E.T * R_N).T
    square = (arr - E) ** 2 / E
    # 期望频数为0时，做除数没有意义，不计入卡方值
    square[ E == 0 ] = 0
    # 卡方值
    v = square.sum()
    return v

def chiMerge(df, col, target, max_groups=None, threshold=None):
    '''
   卡方分箱
   df: pandas dataframe数据集
   col: 需要分箱的变量名（数值型）
   target: 类标签
   max_groups: 最大分组数。
   threshold: 卡方阈值，如果未指定max_groups，默认使用置信度95%设置threshold。
   return: 包括各组的起始值的列表.
   '''

    freq_tab = pd.crosstab(df[ col ], df[ target ])

    # 转成numpy数组用于计算。
    freq = freq_tab.values

    # 初始分组切分点，每个变量值都是切分点。每组中只包含一个变量值.

    # 分组区间是左闭右开的，如cutoffs = [1,2,3]，则表示区间 [1,2) , [2,3) ,[3,3+)。
    cutoffs = freq_tab.index.values

    # 如果没有指定最大分组
    if max_groups is None:
        # 如果没有指定卡方阈值，就以95%的置信度（自由度为类数目-1）设定阈值。
        if threshold is None:
            # 类数目
            cls_num = freq.shape[ -1 ]
            threshold = chi2.isf(0.05, df = cls_num - 1)

    while True:
        minvalue = None
        minidx = None
        # 从第1组开始，依次取两组计算卡方值，并判断是否小于当前最小的卡方
        for i in range(len(freq) - 1):
            v = chi3(freq[ i:i + 2 ])
            if minvalue is None or (minvalue > v):  # 小于当前最小卡方，更新最小值
                minvalue = v
                minidx = i

        # 如果最小卡方值小于阈值，则合并最小卡方值的相邻两组，并继续循环
        if (max_groups is not None and max_groups < len(freq)) or (threshold is not None and minvalue < threshold):
            # minidx后一行合并到minidx
            tmp = freq[ minidx ] + freq[ minidx + 1 ]
            freq[ minidx ] = tmp
            # 删除minidx后一行
            freq = np.delete(freq, minidx + 1, 0)
            # 删除对应的切分点
            cutoffs = np.delete(cutoffs, minidx + 1, 0)

        else:  # 最小卡方值不小于阈值，停止合并。
            break
    return cutoffs

def value2group(x, cutoffs):
    '''
   将变量的值转换成相应的组。
   x: 需要转换到分组的值
   cutoffs: 各组的起始值。
   return: x对应的组，如group1。从group1开始。
   '''

    # 切分点从小到大排序。
    cutoffs = sorted(cutoffs)
    num_groups = len(cutoffs)

    # 异常情况：小于第一组的起始值。这里直接放到第一组。
    # 异常值建议在分组之前先处理妥善。
    if x < cutoffs[ 0 ]:
        return 'group1'

    for i in range(1, num_groups):
        if cutoffs[ i - 1 ] <= x < cutoffs[ i ]:
            return 'group{}'.format(i)

    # 最后一组，也可能会包括一些非常大的异常值。
    return 'group{}'.format(num_groups)

def calWOE(df, var, target):
    '''
   计算WOE编码
   param df：数据集pandas.dataframe
   param var：已分组的列名，无缺失值
   param target：响应变量（0,1）
   return：编码字典
   '''
    eps = 0.000001  # 避免除以0
    gbi = pd.crosstab(df[ var ], df[ target ]) + eps
    gb = df[ target ].value_counts() + eps
    gbri = gbi / gb
    gbri[ 'woe' ] = np.log(gbri[ 1 ] / gbri[ 0 ])
    return gbri[ 'woe' ].to_dict()

def calIV(df, var, target):
    '''
   计算IV值
   param df：数据集pandas.dataframe
   param var：已分组的列名，无缺失值
   param target：响应变量（0,1）
   return：IV值
   '''
    eps = 0.000001  # 避免除以0
    gbi = pd.crosstab(df[ var ], df[ target ]) + eps
    gb = df[ target ].value_counts() + eps
    gbri = gbi / gb
    gbri[ 'woe' ] = np.log(gbri[ 1 ] / gbri[ 0 ])
    gbri[ 'iv' ] = (gbri[ 1 ] - gbri[ 0 ]) * gbri[ 'woe' ]
    return gbri[ 'iv' ].sum()


def box(df, col, target, col_chi):
    cutoffs = chiMerge(df, col, target, max_groups = 8)
    df[ col_chi ] = df[ col ].apply(value2group, args = (cutoffs,))
    return df

columns=['sales_profit', 'maintenance_profit', 'maintenance_revenue_rate', 'vat_burden', 'inventory_turnover', 'cost_profit_rate', 'theoretical_tax_burden', 'total_tax_control',  'single_station_fee',  'premium_return_rate']

for i in columns:
    df=box(df,i,'label',i+'chi')














#特征处理，转变为独热编码

def end_hot():
    from sklearn import preprocessing
    one_hot = preprocessing.OneHotEncoder(sparse=False)
    data_temp=pd.DataFrame(one_hot.fit_transform(df[['sale_type','sale_mode']]),
                 columns=one_hot.get_feature_names(['sale_type','sale_mode']),dtype='int32')
    df1=pd.concat((df,data_temp),axis=1)
    return df1
df1=end_hot()

#数据可视化
plt.rcParams['font.sans-serif'] = 'SimHei'
#观察合规企业

grouped=df1['label'].groupby(df1['label']).count()
print("不合规企业占比：{:.2%}".format(grouped[1]/df1.shape[0]))
grouped.plot(kind='bar')
plt.show()
plt.xticks(rotation = 0)



#可视化主营车型的店与纳税合规


pd.crosstab(df1.sale_type, df1.label).plot(kind = 'bar', stacked = True)
plt.xticks(rotation = 0)
plt.xlabel('Sale Type')
plt.legend([ 'Normal', 'Abnormal' ])
plt.title('SaleType&Tax')
plt.show()

#可视化店铺模式与纳税合规


pd.crosstab(df1.sale_mode, df1.label).plot(kind = 'bar', stacked = True)
temp = pd.crosstab(df1.sale_mode, df1.label)
plt.xticks(rotation = 0)
plt.xlabel('Sale Mode')
plt.legend([ 'Normal', 'Abnormal' ])
plt.title('SaleMode&Tax')
plt.show()



columns = [ 'sales_profitchi', 'maintenance_profitchi', 'maintenance_revenue_ratechi', 'vat_burdenchi',
            'inventory_turnoverchi', 'cost_profit_ratechi', 'theoretical_tax_burdenchi', 'total_tax_controlchi',
            'single_station_feechi', 'premium_return_ratechi' ]
for i in columns:
    pd.crosstab(df1[i], df1.label).plot(kind = 'bar', stacked = True)
    temp = pd.crosstab(df1[i], df1.label)
    plt.xticks(rotation = 45)
    plt.xlabel(i)
    plt.legend([ 'Normal', 'Abnormal' ])
    plt.title(i+'&Tax')
    plt.show()

#查看变量相关性

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df1[ 'sale_type' ] = le.fit(df1[ 'sale_type' ]).transform(df1[ 'sale_type' ])
df1[ 'sale_mode' ] = le.fit(df1[ 'sale_mode' ]).transform(df1[ 'sale_mode' ])
dff_relative=df1.iloc[:,1:16]
matrix = dff_relative.corr()
f, ax = plt.subplots(figsize = (9, 6))
sns.heatmap(matrix, vmax = .8, square = True, cmap = 'BuPu',annot=True)
plt.xticks(rotation = 45)

#model检验

from sklearn.model_selection import train_test_split
df1.drop(['tax_num'],inplace=True,axis=1)
temp=['label','sale_type', 'sale_mode', 'sales_profit', 'maintenance_profit', 'maintenance_revenue_rate', 'vat_burden', 'inventory_turnover', 'cost_profit_rate', 'theoretical_tax_burden', 'total_tax_control', '\nlicensing_rate', 'single_station_fee', '\nagent_insurance_rate', 'premium_return_rate',  'sales_profitchi', 'maintenance_profitchi', 'maintenance_revenue_ratechi', 'vat_burdenchi', 'inventory_turnoverchi', 'cost_profit_ratechi', 'theoretical_tax_burdenchi', 'total_tax_controlchi', 'single_station_feechi', 'premium_return_ratechi', 'sale_type_其它', 'sale_type_卡车及轻卡', 'sale_type_商用货车', 'sale_type_国产轿车', 'sale_type_大客车', 'sale_type_工程车', 'sale_type_微型面包车', 'sale_type_进口轿车', 'sale_mode_4S店', 'sale_mode_一级代理商', 'sale_mode_二级及二级以下代理商', 'sale_mode_其它', 'sale_mode_多品牌经营店']
df1=df1[temp]
df1.replace('group1', 1, inplace = True)
df1.replace('group2', 2, inplace = True)
df1.replace('group3', 3, inplace = True)
df1.replace('group4', 4, inplace = True)
df1.replace('group5', 5, inplace = True)
df1.replace('group6', 6, inplace = True)
df1.replace('group7', 7, inplace = True)
df1.replace('group8', 8, inplace = True)
x=df1.iloc[:,1:]
y=pd.DataFrame(df1.iloc[:,0])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


## build model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression()
model.fit(x_train, y_train)

LogisticRegression(C = 1.0, class_weight = None, dual = False, fit_intercept = True, intercept_scaling = 1,
                   max_iter = 100, multi_class = 'ovr', n_jobs = 1, penalty = 'l2', random_state = 1,
                   solver = 'liblinear', tol = 0.0001, verbose = 0, warm_start = False)
#计算准确率
pred_cv = model.predict(x_test)
# calculate how accurate our preditionsare by calculating the acuracy
accuracy_score(y_test, pred_cv)

#分层交叉验证（stratified k-folds cross validation）
# Validation ...Stratified K Fold
from sklearn.model_selection import StratifiedKFold

# make a cross validation logistic model with stratified 5 folds and make predictions for test dataset
i = 1
kf = StratifiedKFold(n_splits = 5, random_state = 1, shuffle = True)
scorelr = []
for train_index, test_index in kf.split(x, y):
    print('\n{} of kfold {}'.format(i, kf.n_splits))
    xtr, xvl = x.loc[ train_index ], x.loc[ test_index ]
    ytr, yvl = y.loc[ train_index ], y.loc[ test_index ]
    model = LogisticRegression(random_state = 1)
    model.fit(xtr, ytr)
    pred_test = model.predict(xvl)
    score = accuracy_score(yvl, pred_test)
    print('accuracy_score', score)
    scorelr.append(score)
    i += 1
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
i = 1
kf = StratifiedKFold(n_splits = 5, random_state = 1, shuffle = True)
scoredt = [ ]
for train_index, test_index in kf.split(x, y):
    print('\n{} of kfold {}'.format(i, kf.n_splits))
    xtr, xvl = x.loc[ train_index ], x.loc[ test_index ]
    ytr, yvl = y.loc[ train_index ], y.loc[ test_index ]
    model = tree.DecisionTreeClassifier(random_state = 1)
    model.fit(xtr, ytr)
    pred_test = model.predict(xvl)
    score = accuracy_score(yvl, pred_test)
    print('accuracy_score', score)
    scoredt.append(score)
    i += 1
from sklearn.ensemble import RandomForestClassifier

i = 1
kf = StratifiedKFold(n_splits = 5, random_state = 1, shuffle = True)
scorerfc = [ ]
for train_index, test_index in kf.split(x, y):
    print('\n{} of kfold {}'.format(i, kf.n_splits))
    xtr, xvl = x.loc[ train_index ], x.loc[ test_index ]
    ytr, yvl = y.loc[ train_index ], y.loc[ test_index ]
    model = RandomForestClassifier(random_state = 1, max_depth = 10)
    model.fit(xtr, ytr)
    pred_test = model.predict(xvl)
    score = accuracy_score(yvl, pred_test)
    print('accuracy_score', score)
    scorerfc.append(score)
    i += 1
print('LogisticRegression:{}'.format(np.mean(scorelr)))
print('DecisionTreeClassifier:{}'.format(np.mean(scoredt)))
print('RandomForetClassifier:{}'.format(np.mean(scorerfc)))
# Grid Search （网格搜索）
from sklearn.model_selection import GridSearchCV

# 设置max_depth 测试的区间是1-20，间隔是2；
# 设置n_estimators 测试的区间是1-200，间隔为20
paramgrid = {'max_depth': list(range(1, 20, 2)), 'n_estimators': list(range(1, 200, 20))}

grid_search=GridSearchCV(RandomForestClassifier(random_state=1),paramgrid)

from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(x,y, test_size =0.3, random_state=1)

# Fit the grid search model
grid_search.fit(x_train,y_train)
# 选择效果最好的
grid_search.best_estimator_
# 最佳参数max_depth = 1, n_estimators = 81, random_state = 1
i = 1
kf = StratifiedKFold(n_splits = 5, random_state = 1, shuffle = True)
scorerfcn = [ ]
for train_index, test_index in kf.split(x, y):
    print('\n{} of kfold {}'.format(i, kf.n_splits))
    xtr, xvl = x.loc[ train_index ], x.loc[ test_index ]
    ytr, yvl = y.loc[ train_index ], y.loc[ test_index ]
    model = RandomForestClassifier(random_state = 1, max_depth =1, n_estimators = 81)
    model.fit(xtr, ytr)
    pred_test = model.predict(xvl)
    score = accuracy_score(yvl, pred_test)
    print('accuracy_score', score)
    scorerfcn.append(score)
    i += 1
print('RandomForetClassifier:{}'.format(np.mean(scorerfcn)))

#xgboost
from xgboost import XGBClassifier
i = 1
kf = StratifiedKFold(n_splits = 5, random_state = 1, shuffle = True)
scorexgb=[]
for train_index, test_index in kf.split(x, y):
    print('\n{} of kfold {}'.format(i, kf.n_splits))
    xtr, xvl = x.loc[ train_index ], x.loc[ test_index ]
    ytr, yvl = y.loc[ train_index ], y.loc[ test_index ]
    model = XGBClassifier(n_estimators = 41, max_depth = 5)
    model.fit(xtr, ytr)
    pred_test = model.predict(xvl)
    score = accuracy_score(yvl, pred_test)
    print('accuracy_score', score)
    scorexgb.append(score)
    i += 1
# 依旧用之前设置的参数paramgrid，这里不用重新设置
grid_search = GridSearchCV(XGBClassifier(random_state = 1), paramgrid)
# 训练模型
grid_search.fit(x_train, y_train)

grid_search.best_estimator_
#max_depth = 1,n_estimators=21
i = 1
kf = StratifiedKFold(n_splits = 5, random_state = 1, shuffle = True)
scorexgbn = [ ]
for train_index, test_index in kf.split(x, y):
    print('\n{} of kfold {}'.format(i, kf.n_splits))
    xtr, xvl = x.loc[ train_index ], x.loc[ test_index ]
    ytr, yvl = y.loc[ train_index ], y.loc[ test_index ]
    model = XGBClassifier(n_estimators = 21, max_depth = 1)
    model.fit(xtr, ytr)
    pred_test = model.predict(xvl)
    score = accuracy_score(yvl, pred_test)
    print('accuracy_score', score)
    scorexgbn.append(score)
    i += 1

# 总结
# 特征重要性feature_importances_
importances = pd.Series(model.feature_importances_, index = x.columns)
importances.plot(kind = 'barh', figsize = (12, 8))

print('LogisticRegression:{}'.format(np.mean(scorelr)))
print('DecisionTreeClassifier:{}'.format(np.mean(scoredt)))
print('RandomForetClassifier:{}'.format(np.mean(scorerfc)))
print('调参后RandomForetClassifier:{}'.format(np.mean(scorerfcn)))
print('XGB:{}'.format(np.mean(scorexgb)))
print('调参后XGB:{}'.format(np.mean(scorexgbn)))




clf = DecisionTreeClassifier(criterion='gini',max_depth=3,random_state=2)
clf.fit(x_train,y_train)
import dtreeviz
viz = dtreeviz(clf,
               x_train,
               y_train,
               target_name='',
               feature_names=np.array(x.columns),
               class_names=['0','1'])











