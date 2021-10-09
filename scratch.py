import pandas as pd
import numpy as np
from openpyxl import load_workbook
import xlrd
import matplotlib.pyplot as plt
from DateTime import DateTime
from scipy.signal import argrelextrema

"""
data = {"grammer":["Python","C","Java","GO",np.nan,"SQL","PHP","Python"],
       "score":[1,2,np.nan,4,5,6,7,10]}
df = pd.DataFrame(data)
#提取符合条件的字段
df[df['grammer']=='Python']
#修改列名 rename
df.rename(columns = {"score":'popularity'},inplace = True)
#统计出现次数 .value_counts()
df['grammer'].value_counts()
#将空值用上下值的平均值替代 interpolate fillna
df['popularity']=df['popularity'].fillna(df['popularity'].interpolate())
#提取popularity列中值大于3的行
df[df['popularity']>3]
#去除重复 drop_duplicates
df.drop_duplicates(['grammer'])
#计算Popularity平均值
df["popularity"].mean()
#将grammer列转换为list to_list
df['grammer'].to_list()
#将dataframe保存为excel to_excel
# df.to_excel("test.xls")
#查看数据行列数
df.shape
#提取popularity大于3小于7的行
a=df[(df['popularity']>3) &  (df['popularity']<7)]
#交换两列位置
temp=df['popularity']
df.drop(labels=['popularity'],axis=1,inplace = True)
df.insert(0,'popularity',temp)
#提取popularity列最大值所在行
df[df['popularity']==df['popularity'].max()]
#查看最后五行数据
df.tail()
df.iloc[(len(df)-5):len(df),:]
#删除最后一行数据
df.drop([len(df)-1],inplace=True)
#添加一行数据['Perl',6.6]
row={'grammer':'Perl','popularity':'6.6'}
df.append(row,ignore_index = True)
#排序
df.sort_values('popularity',inplace = True)
#统计grammer列每个字符串的长度
df['grammer'].fillna('R',inplace = True)
df['len_string']=list(map(lambda x : len(x),df['grammer']))
"""

"""
#读取本地EXCEL数据
df=pd.read_excel('practice.xls')
#查看前五行
df.head(5)
#将salary数据转换为最大值与最小值的平均值
def func(df):
    temp1=df['salary'].split('-')
    tmin=int(temp1[0].strip('k'))
    tmax = int(temp1[ 1 ].strip('k'))
    df['salary']=int((tmin+tmax)/2*1000)
    return df
df=df.apply(func,axis=1)
#根据学历分组计算平均工资
df.groupby('education').mean('salary')
#将time转换为月日
for i in range(len(df)):
    df.iloc[i,0]=df.iloc[i,0].to_pydatetime().strftime('%m-%d')
#数据索引类型内存信息
# df.info()
#汇总统计
df.describe()
#根据salary分三组
group_names=['low','med','high']
df['group']=pd.cut(df['salary'],3,labels = group_names)
#降序
df=df.sort_values('salary',ascending = False)
#取出对应行数据
df.iloc[5,:]
#计算中位数
np.median(df['salary'])
#绘制salary频率分布图
# df.salary.plot(kind='hist')
#绘制薪资水平密度曲线
# df.salary.plot(kind='kde')
# 删除最后一列categories
# del df['group']
df.drop(columns='group',inplace=True)
#合并前两列
df['test']=df['createTime']+df['education']
#合并指定列
df['test1']=df['education']+df['salary'].map(str)
#计算salary最大值与最小值的差
df[['salary']].apply(lambda x:x.max()-x.min())
#将最后一行与第一行拼接
pd.concat([df[:1], df[-2:-1]])
#将第8行数据添加至末尾
df=df.append(df.iloc[7])
#设置索引列
df.set_index('createTime')
#生成和df长度相同的随机数dataframe
df1=pd.DataFrame(np.random.randint(1,9999,21))
#合并df1与df
df=df.reset_index()
df1=df1.reset_index()
df=pd.concat((df,df1),axis = 1)
del df['index']
#生产新的一列 salary-df1
df['test2']=df['salary']-df[0]
#检查数据中是否含有任何缺失值
df.isnull().values.any()
#将salary列类型转换为浮点数
df['salary']=df['salary'].astype(float)
#计算salaries大于10000的次数
df[df['salary']>10000].count()
#查看每种学历出现的次数
df['education'].value_counts()
#查看education列共有几种学历
df['education'].nunique()
#50.提取salary与new列的和大于60000的最后3行
df['test2_temp']=df["salary"]+df['test2']
df['test2_temp']=df['test2_temp'].astype(int)
temp=df[df['test2_temp']>40000]
temp1=temp.iloc[-3:]
print(temp1)
del temp
del temp1
del df1
del group_names
"""

"""
#51.使用绝对路径读取本地Excel数据

data=pd.read_csv(r"")

#查看每列数据缺失值情况
data.isnull().sum()
#提取日期列含有空值的行
data[data['Trddt'].isnull()]
#输出每列缺失值具体行数
# for columnnames in data.columns:
#     if data[columnnames].count()!=len(data):
#         row=data[columnnames][data[columnnames].isnull().values==True].index.tolist()
#         print('列名："{}", 第{}行位置有缺失值'.format(columnnames,row))
#.删除所有存在缺失值的行
data.dropna(axis=0,inplace = True)
#绘制收盘价的折线图

# plt.plot(data['Clsprc'])
# 同时绘制开盘价与收盘价
# data[['Clsprc','Opnprc']].plot()
# 绘制涨跌幅的直方图
# data['涨跌幅(%)']=(data['Clsprc']-data['Opnprc'])/data['Opnprc']
# data['涨跌幅(%)'].hist(bins=20)
# .以data的列名创建一个dataframe
temp=pd.DataFrame( data.columns.to_list())
#打印所有换手率不是数字的行
temp=data[data['Dretwd']!=0.01]
# temp=data[data['Dsmvtll'].isin('sss')]
#重置data的行号
data=data.reset_index()
# k=[]
# for i in range(len(data)):
#     if type(data.iloc[i,5])!=float:
#         k.append(i)
# data.drop(labels=k,inplace=True)

#绘制换手率的密度曲线
# data['Dretwd'].plot(kind='kde')
del data['index']
#计算前后的差
data['Clsprc_diff']=data['Clsprc'].diff()
#计算前后的差率
data['Clsprc_pct_change']=data['Clsprc_diff']/data['Clsprc']
data = data.set_index('Trddt')
# 以5个数据作为一个数据滑动窗口，在这个5个数据上取均值(收盘价)
data['Clsprc'].rolling(5).mean()
# 将收盘价5日均线、20日均线与原始数据绘制在同一个图上
# data['Clsprc'].plot()
# data['Clsprc'].rolling(5).mean().plot()
# data['Clsprc'].rolling(20).mean().plot()
# 按周为采样规则，取一周收盘价最大值
data.index = pd.to_datetime(data.index)
data['Clsprc'].resample('w').max()
# 绘制重采样数据与原始数据
# data['Clsprc'].plot()
# data['Clsprc'].resample('w').max().plot()
# 将数据往后移动5天
data.shift(5)
# 使用expending函数计算开盘价的移动窗口均值
data['Clsprc'].expanding(min_periods = 1).mean()
# 绘制上一题的移动均值与原始数据折线图
# data['Clsprc'].expanding(min_periods = 1).mean().plot()
# data['Clsprc'].plot()
# 计算布林指标
data['移动平均']=data['Clsprc'].rolling(20).mean()
data['下限']=data['移动平均']-2*data['Clsprc'].rolling(20).std()
data['上限']=data['移动平均']+2*data['Clsprc'].rolling(20).std()
# 计算布林线并绘制
# data[['Clsprc','下限','上限']].plot()
"""

"""
# 81.从NumPy数组创建DataFrame
rt1=np.random.randint(1, 50000, 10000)
rt1=pd.DataFrame(rt1)

rt2=np.arange(0,50000,5)
rt2=pd.DataFrame(rt2)

rt3=np.random.normal(0,1,10000)
rt3=pd.DataFrame(rt3)

df=pd.concat([rt1,rt2,rt3],axis = 1)
# 查看df所有数据的最小值、25%分位数、中位数、75%分位数、最大值
df.describe()
df.columns=['col1','col2','col3']
# 提取第一列中不在第二列出现的数字
df['col1'][~df['col1'].isin(df['col2'])]
# 提取第一列和第二列出现频率最高的三个数字
# temp=df['col1'].append(df['col2'])
# temp1=temp.value_counts()
# print(temp1[:3])
# 提取第一列中可以整除5的数字位置
np.argwhere((df['col1']%5==0).values)
# .计算第一列数字前一个与后一个的差值
df.diff()
#将col1,col2,clo3三列顺序颠倒
df55=df.iloc[:,::-1]
df44=df.iloc[::-1,:]
del [df55,df44,rt1,rt2,rt3]
# 提取第一列位置在1,10,15的数字
# df1=df.iloc[[1,10,15],0]
# 查找第一列的局部最大值位置
# x=np.array(df['col1'])
# y=argrelextrema(x,lambda a,b:a>b)
# 按行计算df的每一行均值
df.mean(axis=1)
# 97.对第二列计算移动平均值
df['col2'].rolling(3).mean()
# 98.将数据按照第三列值的大小升序排列
df.sort_values('col3',inplace = True)
#将第一列大于50的数字修改为'高'
df.col1[df['col1']>5550]='high'
# 计算第二列与第三列之间的欧式距离
np.linalg.norm(df['col2']-df['col3'])
"""

data=pd.read_csv(r"C:\Users\Aa774\Desktop\日个股回报率文件171058276\TRD_Dalyr.csv",converters={'Opnprc': lambda x: '高' if float(x) > 5 else '低'})
data.iloc[::20][['Opnprc']]
df=pd.DataFrame(np.random.random(10)**10,columns = ['data'])
df=df.round(3)
f1=lambda x:'%.2f%%' %(x*100)
df['de']=df['data'].apply(f1)
df['data'].argsort()[::-1][7]
df=df.iloc[::-1,:]
df1= pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
'key2': ['K0', 'K1', 'K0', 'K1'],
'A': ['A0', 'A1', 'A2', 'A3'],
'B': ['B0', 'B1', 'B2', 'B3']})

df2= pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
'key2': ['K0', 'K0', 'K0', 'K0'],
'C': ['C0', 'C1', 'C2', 'C3'],
'D': ['D0', 'D1', 'D2', 'D3']})

df=pd.merge(df1,df2,how='right',on=['key1','key2'])

np.where(data.Hiprc==data.Loprc)

np.argwhere((data['Hiprc']>data['Hiprc'].mean()).values)[2]

data[['Hiprc']].apply(np.sqrt)

data['split']=data['Trddt'].str.split('/')

data=data.dropna()

data=data[data['Trddt'].str.startswith('2020')]

data1=pd.pivot_table(data,values = ['Hiprc','Loprc'],index='Trddt')


