# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 11:38:51 2017

@author: ly
"""
import sys
sys.path.append(r"C:\Users\ly\Desktop")

import numpy as np
import pandas as pd
from pyecharts import Pie
import matplotlib.pyplot as plt
plt.style.use('ggplot')  #风格设置近似R这种的ggplot库
import seaborn as sns
sns.set_style('whitegrid')
import missingno as msno
from scorecard import iv

%matplotlib inline

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
# 忽略弹出的warnings
import warnings
warnings.filterwarnings('ignore')  

pd.set_option('display.float_format', lambda x: '%.5f' % x)

data =pd.read_csv(r"F:\python\python\Credit\LoanStats_2016Q1\train.csv",encoding="gbk",index_col =0,low_memory=False)
data.head()
#ac=data.head(10000)
#ac.to_excel(r"F:\python\python\Credit\LoanStats_2016Q1\train_.xlsx")
''' 三期以前逾期的客户暂时定义为 欺诈客户 '''
acc = data[data["last_pymnt_amnt"]==0]
pd.value_counts(acc["last_pymnt_amnt"])

check_null = data.isnull().sum(axis=0).sort_values(ascending=False)/float(len(data)) #查看缺失值比例
print(check_null[check_null > 0.2]) # 查看缺失比例大于20%的属性。

thresh_count = len(data)*0.4 # 设定阀值
data = data.dropna(thresh=thresh_count, axis=1 ) #若某一列数据缺失的数量超过阀值就会被删除
data =data.dropna( thresh=20 ) 

#同质性剔除 因为样本数据含有字符型数据，无法使用sklearn中的方差选择
#变量的最大占比是否大于>90%
def value_ratio(df,ratiolimit = 0.9):
    recordcount= len(df)
    x = [ ]
    for col in df.columns:
        primaryvalue = df[col].value_counts().index[0]
        ratio = float(df[col].value_counts().iloc[0])/recordcount
        x.append([ratio,primaryvalue])       
    feature_primaryvalue_ratio = pd.DataFrame(x,index =df.columns)
    feature_primaryvalue_ratio.columns = ['primaryvalue_ratio','primaryvalue']
    needcol = feature_primaryvalue_ratio[feature_primaryvalue_ratio['primaryvalue_ratio']<=ratiolimit]
    needcol = list(needcol.index)
    if len(needcol)>0:                
        selected_data = df[needcol]
    return selected_data         
#剔除20个同质变量
data = value_ratio(data).drop(["last_pymnt_amnt","total_rec_prncp","out_prncp","out_prncp_inv"
           ,"total_pymnt_inv","total_pymnt","funded_amnt","funded_amnt_inv"],axis=1)

#储存好已经初步处理的数据
data =data.reset_index(drop=True)
data.to_csv(r"F:\python\python\Credit\LoanStats_2016Q1\train_one.csv")























