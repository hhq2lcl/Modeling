# -*- coding: utf-8 -*-
"""
Created on Wed May  9 11:50:38 2019
@author: ts_data
"""

import sys
sys.path.append(r"C:\Users\hhq\Desktop")
from scorecard import binn,model_test,make_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression#学习速率(步长)参数，权重的正则化参数 
import scipy.stats.stats as stats
import statsmodels.api as sm

''' 1. 导入数据,预处理 '''
# =============================================================================
# 建模前数据准备在sas中已处理
# 名义变量进行了01 转换
# 代码如下
# =============================================================================

data= pd.read_csv(r"F:\data\model\01_Dataset\01_original\sample_data.csv",encoding="utf-8")

# =============================================================================
# 对建模目标值进行分组
# target
# 0    2233
# 1     374
# 代码如下
# =============================================================================

data.groupby("target")["target"].count()

# =============================================================================
# 对建模变量进行同质化处理
# 由于模型需要，将y标签移到最后一列，才能符合模型，放前面跑不了
# 先调整y标签位置，再进行同质化处理，同质化处理函数已包装好
# 同质化检查，变量由108变为69个
# 代码如下
# =============================================================================

##将y标签移到最后一列，才能符合模型，放前面跑不了
y=data["target"]
x=data.drop("target",axis=1)
data1=x
data1.insert(108, 'y', y)

def select_primaryvalue_ratio(data, ratiolimit = 0.95):
    '''按照命中率进行筛选,首先计算每个变量的命中率,这个命中率是指维度中占比最大的值的占比  '''     
    recordcount = data.shape[0]
    x = []
    #循环每一个列，并取出出现频率最大的那个值;index[0]是取列名,iloc[0]是取列名对应的值
    #df1.selfquery_in3m.value_counts() 是按分值占比的从大到小的顺序排
    #df1.selfquery_in3m.value_counts().index[0] 是按分值占比的从大到小的顺序排取最大的值
    for col in data.columns:
        primaryvalue = data[col].value_counts().index[0]
        ratio = float(data[col].value_counts().iloc[0])/recordcount
        x.append([ratio,primaryvalue])       
    feature_primaryvalue_ratio = pd.DataFrame(x,index = data.columns)
    feature_primaryvalue_ratio.columns = ['primaryvalue_ratio','primaryvalue']
        
    needcol = feature_primaryvalue_ratio[feature_primaryvalue_ratio['primaryvalue_ratio']<ratiolimit]
    needcol = needcol.reset_index()
    select_data = data[list(needcol['index'])]
    return select_data, feature_primaryvalue_ratio  

data2, feature_primaryvalue_ratio =select_primaryvalue_ratio(data1,ratiolimit = 0.80)

''' 2. 变量区分，分别处理 '''

# =============================================================================
# 名义型变量处理-连续型变量处理
# 名义型变量处理已在sas中做处理
# 连续型变量处理均等分箱
# 名义型变量视为x_categ，连续型变量处理x_num
# 代码如下
# =============================================================================

car_list = ['DESIRED_PRODUCT_G',
'DESIRED_LOAN_AMOUNT_G',
'MARRIAGE_G',
'CHILD_COUNT_G',
'AGE_G',
'EDUCATION_G',
'LOCAL_RES_YEARS_G',
'PROPERTY_HOUSING_G',
'IS_HAS_HOURSE_G',
'IS_HAS_HOURSE_CAR_G',
'INDUSTRYS_NAME',
'WORK_YEARS_G',
'OC_G',
'PROPERTY_WORKING_G',
'YEARLY_INCOME_G',
#'SALARY_DISTRIBUTION_G',
#'IS_HAS_SOCIAL_HOUSING_G',
# 说明：由于基本信息变量的先在SAS中处理，故同质化处理主要减少的是征信变量，IS_HAS_SOCIAL_HOUSING_G、SALARY_DISTRIBUTION_G
'IS_HAS_SOCIAL_HOUSING_H_G',
]
x_categ = x[car_list]
x_num = data2.drop(car_list,axis=1)

''' 2.1 名义型变量处理 '''
# =============================================================================
# 名义型变量处理IV值
# 代码如下
# =============================================================================

iv_detail1 = binn.class_iv(x_categ,y)
col_mapping={"bad":"BAD","good":"GOOD"}
iv_detail1=iv_detail1.rename(columns=col_mapping,copy=False)
iv_detail1.to_csv(r"C:\Users\hhq\Desktop\iv1.csv")
iv_detail1.to_excel(r"C:\Users\hhq\Desktop\iv1.xlsx")

''' 2.2 连续型变量处理 '''
# =============================================================================
# 连续型变量处理IV值
# 代码如下
# =============================================================================

iv_detail2 = binn.cal_iv(pd.concat((x_num,y),axis=1),group=5)
#1个0.09
#iv_detail2 = binn.cal_iv(pd.concat((x_num,y),axis=1),group=6)
#0.10最高，2个0.09以上
#iv_detail2 = binn.cal_iv(pd.concat((x_num,y),axis=1),group=7)
#0.10最高，2个0.09以上
#iv_detail2 = binn.cal_iv(pd.concat((x_num,y),axis=1),group=8)
#0.10最高，1个0.09以上
#iv_detail2 = binn.cal_iv(pd.concat((x_num,y),axis=1),group=20)
#0.10最高，2个0.10以上
#iv_detail2 = binn.cal_iv(pd.concat((x_num,y),axis=1),group=10)
#0.10最高，2个0.09以上

iv_detail2.to_csv(r"C:\Users\ts_data\Desktop\iv.csv")

select_feat= iv_detail2[iv_detail2["ori_IV"]>0.02]["var_name"].drop_duplicates()
X=x[select_feat]

acorr=X.corr()
b=-1
for i in acorr.index:
    rowdata = acorr.ix[i,:]
    b = b + 1
    if any(np.abs(rowdata[:b]) >= 0.5):#绝对值有一项>0.75，则运行
        acorr=acorr.drop(i)
        acorr=acorr.drop(i,axis=1)
        b=b-1
acorr.index



''' 2.3 手动修改IV'分布- 并作为最终输出 '''

WOE_bestsplit1 = pd.read_excel(r"F:\data\model\02_DataProcess\03_best_IV\bestsplit.xlsx")
WOE_bestsplit1 = WOE_bestsplit1.sort_values(by=['ori_IV','var_name','max'],ascending=[False,True,True])

select_feat1= WOE_bestsplit1["var_name"].drop_duplicates()  
X_s=x[select_feat1]


X_ms = pd.concat([X_s,x_categ],axis=1)

WOE_detail1 =pd.concat([WOE_bestsplit1,iv_detail1],axis=0)
#WOE_detail1.to_excel(r"F:\data\model\01_Dataset\02_Interim\WOE_detail_1.xlsx")


ac=X_ms.corr()
b=-1
for i in ac.index:
    rowdata = ac.ix[i,:]
    b = b + 1
    if any(np.abs(rowdata[:b]) >= 0.5):#绝对值有一项>0.75，则运行
        ac=ac.drop(i)
        ac=ac.drop(i,axis=1)
        b=b-1
ac.index
X_ms=X_ms[ac.index]


new_x=pd.DataFrame()
for i in X_ms.columns:
    print(i)
    new = binn.applyBinwoe(X_ms[i],WOE_detail1[WOE_detail1["var_name"]==i])
    
    new_x=pd.concat([new_x,new],axis=1)

X_ms=new_x


''' 2.4 测试数据 '''

test =pd.read_csv(r"F:\data\model\01_Dataset\01_original\test_sample.csv",encoding="utf-8")

test_x = test.drop("target",axis=1)
test_y = test["target"]
test_x=test[X_ms.columns]
new_x=pd.DataFrame()
for i in X_ms.columns:
    print(i)
    new = binn._applyBinwoe(test_x[i],WOE_detail1[WOE_detail1["var_name"]==i])
    new_x=pd.concat([new_x,new],axis=1)

test_x=new_x



''' 3 建立模型 '''

clf1 = LogisticRegression(penalty='l2',class_weight="balanced")
clf1.fit(X_ms,y)


''' 4 模型评估 '''

model_test.ROC_plt(clf1,X_ms,y)
model_test.KS_plt(clf1,X_ms,y,ksgroup=10)


model_test.ROC_plt(clf1,test_x,test_y)
model_test.KS_plt(clf1,test_x,test_y,ksgroup=10)


#模型系数
formula=model_test.get_lr_formula(clf1,X_ms)
#评分卡分值
scorecard = make_score.make_scorecard(formula,WOE_detail1)


#训练集数据，模型系数，评分卡导出excel
writer = pd.ExcelWriter(r"F:\data\model\01_Dataset\02_Interim\model_1.xlsx")
scorecard.to_excel(writer,sheet_name='scorecard')
formula.to_excel(writer,sheet_name='formula')


aa={}
b={}

for i in scorecard.var_name.drop_duplicates():
    a=scorecard[scorecard["var_name"]==i].set_index("woe").T.to_dict("records")
    aa[i]=a[5]
    
X_fs_score = X_ms.replace(aa)

X_fs_score["score"] =X_fs_score.sum(axis=1)

data_1 = pd.concat([X_fs_score,y],axis=1)
#data_1.to_excel(r"F:\data\model\01_Dataset\02_Interim\train_1.xlsx")


#测试集数据，模型系数，评分卡导出Excel
aa={}
b={}

for i in scorecard.var_name.drop_duplicates():
    a=scorecard[scorecard["var_name"]==i].set_index("woe").T.to_dict("records")
    aa[i]=a[5]
    
test_x_score = test_x.replace(aa)

test_x_score["score"] =test_x_score.sum(axis=1)

data_2 = pd.concat([test_x_score,test_y],axis=1)
#data_2.to_excel(r"F:\data\model\01_Dataset\02_Interim\test_1.xlsx")



#模型分组情况
train_KS=make_score.score_ks(data_1[["score","target"]],1) 

test_KS=make_score.score_ks(data_2[["score","target"]],1) 

print(train_KS["score_KS"],test_KS["score_KS"])

''' 5 模型检测 '''

test_y = test["target"]
test_x=test[X_ms.columns]
new_x=pd.DataFrame()
for i in test_x.columns:
    print(i)
    new = binn._applyBinwoe(test_x[i],scorecard[scorecard["var_name"]==i],"rank")
    new_x=pd.concat([new_x,new],axis=1)

test_iv= binn.class_iv(new_x,test_y)
test_iv.to_excel(r"F:\data\model\01_Dataset\02_Interim\test_iv1.xlsx")
