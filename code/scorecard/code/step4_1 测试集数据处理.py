# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 14:33:03 2018

@author: ly
"""


import pandas as pd
import sys
sys.path.append(r"C:\Users\ly\Desktop")
from scorecard import bining


test_data =pd.read_csv(r"F:\python\python\Credit\LoanStats_2016Q1\test.csv",encoding="gbk",index_col =0,low_memory=False)
thresh_count = len(test_data)*0.4 # 设定阀值
test_data = test_data.dropna(thresh=thresh_count, axis=1 ) #若某一列数据缺失的数量超过阀值就会被删除
test_data =test_data.dropna( thresh=20 ) 

test_data =test_data.reset_index(drop=True)



test_objectcolumns = test_data.select_dtypes(include=["object"]).columns
test_data[test_objectcolumns] = test_data[test_objectcolumns].fillna("Unknown") #以分类“Unknown”填充缺失值

test_numColumns = test_data.select_dtypes(include=["float64"]).columns
test_data[test_numColumns] = test_data[test_numColumns].fillna(0)


test_data["y"] = test_data["target"].replace({'Good':0,'Bad':1})
test_data["home_ownership"]=test_data["home_ownership"].replace({"ANY":"MORTGAGE"})
test_data["grade"]=test_data["grade"].replace({"E":"E","F":"E","G":"E"})




test_y = test_data["y"]


new_x=pd.DataFrame()
for i in X_ms.columns:
    print(i)
    new = bining._applyBinwoe(test_data[i],woe[woe["var_name"]==i])
    
    new_x=pd.concat([new_x,new],axis=1)
test_x=new_x





