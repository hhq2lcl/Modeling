# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 14:59:28 2018

@author: ly
"""
import pandas as pd 

data1.select_dtypes(include=[np.number]).isnull().sum().sort_values(ascending=False)/float(len(data))

numColumns = data1.select_dtypes(include=["float64"]).columns
data1[numColumns] = data1[numColumns].fillna(0)

# 连续变量处理方法1 连续变量的相关性
corr1=data1[numColumns].corr()
cul1={}
b=0
for i in corr1.index:
    a=[]
    rowdata = corr1.ix[i,:]

    print (i)
    if any(np.abs(rowdata) >= 0.6):#绝对值有一项>0.75，则运行
        print (i)
        b = b + 1
        cul1[b]=list(rowdata[np.abs(rowdata)>0.6].index)
        corr1=corr1.drop(rowdata[np.abs(rowdata)>0.6].index,axis=1)

ac.index

cul_iv=pd.DataFrame()
for i in cul1:
    print (cul1[i])
    cul_woe= iv.cal_iv(pd.concat([data1[cul1[i]],y],axis=1),group=5)
    best_ = list(cul_woe["var_name"].drop_duplicates())[0]
    cul_iv=pd.concat([cul_iv,cul_woe[cul_woe["var_name"]==best_]],axis=0)

num_iv_2=cul_iv

#连续变量处理方法2
num_iv = iv.cal_iv(pd.concat([data1[numColumns],y],axis=1),group=5)

#acc=pd.DataFrame()
#acc["open_il_12m"]=data1["open_il_12m"]
#acc["open_il_24m"] = data1["open_il_24m"]/data1["open_il_12m"]
#acc["open_il_24m"]=acc["open_il_24m"].fillna(0)
#acc.ix[acc["open_il_24m"]==np.inf ,"open_il_24m"] =-1
#acc_iv = iv.cal_iv(pd.concat([acc["open_il_24m"],y],axis=1),group=10)


num_iv_2 = num_iv[num_iv["ori_IV"]>0.02]

num_sel_feat= num_iv[num_iv["ori_IV"]>0.02]["var_name"].drop_duplicates()
x_num=data1[num_sel_feat]

num_iv_2.to_csv(r"F:\python\python\Credit\LoanStats_2016Q1\num_iv_2.csv")
