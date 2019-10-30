# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 17:01:39 2018

@author: ly
"""

import pandas as pd 
import os


files = os.listdir(r'F:\python\python\Credit\LoanStats_2016Q1\train')
fs = pd.DataFrame()
for f in files:
    t = pd.read_csv(r'F:\python\python\Credit\LoanStats_2016Q1\train\\'+f, encoding='latin-1',skiprows = 1,low_memory=False)
    print(pd.value_counts(t["loan_status"]))

    fs=pd.concat([fs,t],axis=0)
fs = fs.reset_index(drop=True)
fs.to_csv(r"F:\python\python\Credit\LoanStats_2016Q1\train.csv")


data =pd.read_csv(r"F:\python\python\Credit\LoanStats_2016Q1\train.csv",encoding="gbk",index_col =0,low_memory=False)
data =data.dropna( thresh=20 ) 

#储存好已经初步处理的数据
data =data.reset_index(drop=True)
print(pd.value_counts(data["loan_status"]))

print(pd.value_counts(data["issue_d"]))

data["issue_d_T"]=pd.to_datetime(data["issue_d"])

print(pd.value_counts(data["last_pymnt_d"]))

data["last_pymnt_d_T"]=pd.to_datetime(data["last_pymnt_d"])

 
def month_differ(x):
    """暂不考虑day, 只根据month和year计算相差月份
    Parameters
    Return
    ------
    differ: 月份
    """
    month_differ = abs(x.year* 12 + x.month * 1)
    return month_differ
 
 
data["mob"] = data["last_pymnt_d_T"].apply(lambda x:abs(x.year* 12 + x.month * 1))-data["issue_d_T"].apply(lambda x:abs(x.year* 12 + x.month * 1))
print(pd.value_counts(data["mob"]))

data["target"] = "Indet"
data.ix[((data["mob"]>=11) & (data["loan_status"]=="Current")),"target"] ="Good"
data.ix[((data["mob"]>=6) & (data["loan_status"]=="Fully Paid")),"target"] ="Good"
data.ix[((data["mob"]>=2) & (data["loan_status"]=="Late (31-120 days)")),"target"] ="Bad"
data.ix[((data["mob"]>=2) & (data["loan_status"]=="Charged Off")),"target"] ="Bad"

print(pd.value_counts(data["target"]))

data_1 = data[data["target"] !="Indet"]
data_1.groupby(["target","issue_d"])["target"].count()
data_2=data_1.drop(["last_pymnt_d_T","issue_d_T","mob"],axis=1)
test = data_2[data_1["issue_d"]=="Mar-2017"]
print(pd.value_counts(test["target"]))

train =data_2.drop(test.index)

test.to_csv(r"F:\python\python\Credit\LoanStats_2016Q1\test.csv")
train.to_csv(r"F:\python\python\Credit\LoanStats_2016Q1\train.csv")



