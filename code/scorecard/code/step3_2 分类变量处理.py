# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 14:56:51 2018

@author: ly
"""

import pandas as pd 



#类别型变量处理
#缺失值识别
objectcolumns = data1.select_dtypes(include=["object"]).columns
data1[objectcolumns].isnull().sum().sort_values(ascending=False)
'''
next_pymnt_d           29640
emp_title              10358
revol_util               107
last_credit_pull_d         5
title                      4
zip_code                   1
last_pymnt_d_T             0
verification_status        0
int_rate                   0
grade                      0
sub_grade                  0
emp_length                 0
home_ownership             0
purpose                    0
issue_d                    0
loan_status                0
issue_d_T                  0
addr_state                 0
earliest_cr_line           0
initial_list_status        0
last_pymnt_d               0
term                       0
'''


data1["mo_old_loan"] = pd.to_datetime(data1["issue_d_T"]) - pd.to_datetime(data1["earliest_cr_line"])
data1["mo_old_loan"]=data1["mo_old_loan"].apply(lambda x:round(x/np.timedelta64(30,"D")))
#
data1['int_rate'] = data1['int_rate'].str.rstrip('%').astype('float')
data1['revol_util'] = data1['revol_util'].str.rstrip('%').astype('float')
data1=data1.drop(["last_pymnt_d","issue_d_T","earliest_cr_line","next_pymnt_d","last_pymnt_d_T","last_credit_pull_d"],axis=1)
#单独填充
objectcolumns = data1.select_dtypes(include=["object"]).columns

data1[objectcolumns] = data1[objectcolumns].fillna("Unknown")


'''
next_pymnt_d  剔除
title  基本没有预测能力

'''

pd.value_counts(data1["emp_title"])
'''
["term","grade","emp_length","home_ownership","verification_status","purpose","emp_title","sub_grade",
                          "zip_code" ,"title","zip_code","addr_state","initial_list_status"]
'''
object_iv = iv.class_iv(data1[["term","grade","emp_length","home_ownership","verification_status","purpose"
                          ,"initial_list_status"]],y)
object_iv_2 = object_iv[object_iv["ori_IV"]>0.02]
obj_sel_feat= object_iv_2["var_name"].drop_duplicates()

'''
变量处理：

'''
x_object=data1[obj_sel_feat]
x_object["home_ownership"]=x_object["home_ownership"].replace({"ANY":"MORTGAGE"})
x_object["grade"]=x_object["grade"].replace({"E":"E","F":"E","G":"E"})

object_iv = iv.class_iv(x_object,y)


msno.matrix(data[objectColumns]) #缺失值可视化
 
msno.heatmap(data[objectColumns]) #查看缺失值之间的相关性
