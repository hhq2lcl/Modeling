# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 14:55:45 2018

@author: ly
"""

import pandas as pd 

'''
EDA 探索性数据分析
'''
data =pd.read_csv(r'F:\python\python\Credit\LoanStats_2016Q1\train_one.csv',encoding="gbk",index_col =0,low_memory=False)

#因变量分析

data["y"] = data["target"].replace({'Good':0,'Bad':1})

print( '\nAfter Coding:')

pd.value_counts(data["y"])


attr = ["正常", "违约"]
pie = Pie("贷款状态占比")
pie.add("", attr, [int(i) for i in pd.value_counts(data["y"])] ,is_label_show=True)
pie
y=data["y"]
data = data.drop("target",axis=1)




data1.dtypes.value_counts()
'''
float64    58
object     19
int64       1
'''
#2贷款金额分布
plt.figure(figsize=(10, 5))
sns.set()
sns.set_context("notebook", font_scale=1, rc={"lines.linewidth":2 } )
sdisplot_loan = sns.distplot(data['loan_amnt'] )
plt.xticks(rotation=90)
plt.xlabel('Loan amount')
plt.title('Loan amount\'s distribution')

# 贷款期限占比可视化
attr = ["36个月", "60个月"]
pie = Pie("贷款期限占比")
pie.add("", attr, [float(i) for i in pd.value_counts(data["term"])] ,is_label_show=True)
pie


data1.to_csv(r"F:\python\python\Credit\LoanStats_2016Q1\train_two.csv")
