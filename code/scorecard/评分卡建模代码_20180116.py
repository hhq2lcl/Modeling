# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 11:24:39 2017

@author: ly
"""

import sys
sys.path.append(r"C:\Users\ts_data\Desktop")
from scorecard import binn,model_test,make_score



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression#学习速率(步长)参数，权重的正则化参数 
import scipy.stats.stats as stats
import statsmodels.api as sm

'''
建模
'''
"""---------------------------1.导入数据,预处理---------------------------------"""
'''  样本在sas中已经做了异常值，缺失值处理，名义变量进行了01 转换  '''
data= pd.read_csv(r"C:\Users\ts_data\Desktop\sample_data.csv",encoding="utf-8")
data["YEAR_INCOME"] =(data["MONTHLY_SALARY"]+data["MONTHLY_OTHER_INCOME"])*12

#data["YEAR_CINCOME"] =(data["MONTHLY_SALARY"]+data["MONTHLY_OTHER_INCOME"]-data["MONTHLY_EXPENSE"])*12
data.groupby("target")["target"].count()

"""---------------------------1.1同质性剔除--------------------------------"""
'''同质性剔除 如果样本数据含有字符型数据，也可以使用sklearn中的方差选择
# 变量的最大占比是否大于>90%'''
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

data1= value_ratio(data)

y=data["target"]
x=data1.drop("target",axis=1)


#绘制直方图
data['AGE_G'].dropna().hist(bins=20,range=(20,65),alpha=1) 
plt.show()


"""---------------------------2.变量区分，分别处理---------------------------------"""
"""---------------------------2.1名义型变量处理---------------------------------"""
''' '''
car_list = ["EDUCATION","MARRIAGE","GENDER1","IS_HAS_HOURSE1","IS_HAS_CAR1",
            "IS_LIVE_WITH_PARENTS1","SALARY_PAY_WAY1"
            ,"LOCAL_RESCONDITION_G","position_G","comp_type","nonlocal","asset",
            "DESIRED_PRODUCT1","group_Level_g"]
x_categ = x[car_list]
x_num = x.drop(car_list,axis=1)

all_iv_detail = binn.class_iv(x_categ,y)

'''
if 房产性质="公积金按揭购房"  then LOCAL_RESCONDITION_G =0;else if 房产性质="公司宿舍"  then LOCAL_RESCONDITION_G =1;
else if 房产性质="亲属住房"  then LOCAL_RESCONDITION_G =2;else if 房产性质="商业按揭房"  then LOCAL_RESCONDITION_G =3;
else if 房产性质="无按揭购房"  then LOCAL_RESCONDITION_G =4;else if 房产性质="自建房"  then LOCAL_RESCONDITION_G =5;
else if 房产性质="租用"  then LOCAL_RESCONDITION_G =6;else if 房产性质="其他"  then LOCAL_RESCONDITION_G =7;
合并：
4=0（无按揭购房）2=1（亲属住房） 0,3,5 =2 （其他方式购房） 1,6,7=3（租用）

'''
x_categ.LOCAL_RESCONDITION_G=x_categ.LOCAL_RESCONDITION_G.replace({0:0,2:0,3:0,6:2,7:1,1:2,4:0,5:1})

'''
if 单位性质 ="个体" then comp_type=0;else if 单位性质 ="国有股份" then comp_type=2;
else if 单位性质 ="合资企业" then comp_type=1;else if 单位性质 ="机关事业单位" then comp_type=4;
else if 单位性质 ="民营企业" then comp_type=5;else if 单位性质 ="社会团体" then comp_type=6;
else if 单位性质 ="私营企业" then comp_type=7;else if 单位性质 ="外资企业" then comp_type=3;
合并：
  0,4,6,7 =0     1,2=1    3,5=2

'''
x_categ.comp_type=x_categ.comp_type.replace({0:0,1:1,2:0,3:1,4:0,5:1,6:0,7:0})
'''
if  教育程度="硕士及其以上" then  EDUCATION=0;if 教育程度="大学本科" then  EDUCATION=1;
if  教育程度="专科" then EDUCATION=2;if  教育程度="中专" then EDUCATION=3;
if 教育程度="高中" then EDUCATION=4;if  教育程度="初中" then EDUCATION=5;
if 教育程度="小学" then  EDUCATION=6;
合并：

'''
x_categ.EDUCATION=x_categ.EDUCATION.replace({0:0,1:0,2:1,3:1,4:1,5:1,6:1})


'''
if 财产信息 = "有房有车" then asset = 3;else if 财产信息 = "有房无车" then asset =2;
else if 财产信息 = "有车无房" then asset = 1;else if 财产信息 = "无车无房" then asset = 0;
合并：
0,=0  1,2=1 3=2
'''
#第一种 合并

'''第二种'''
x_categ["GM"] = 0
x_categ.ix[((x_categ["GENDER1"]==0) & (x_categ["MARRIAGE"]==1)),["GM"]]=0 
x_categ.ix[((x_categ["GENDER1"]==0) & (x_categ["MARRIAGE"]!=1)),["GM"]]=1 
x_categ.ix[((x_categ["GENDER1"]==1) & (x_categ["MARRIAGE"]==0)),["GM"]]=0 

x_categ["localas"] = 0
x_categ.ix[((x_categ["nonlocal"]==1) & (x_categ["asset"]==0)),["localas"]]=1 
x_categ.ix[((x_categ["nonlocal"]==1) & (x_categ["asset"]==2)),["localas"]]=0 
x_categ.ix[((x_categ["nonlocal"]==0) & (x_categ["asset"]==0)),["localas"]]=1
x_categ.ix[((x_categ["nonlocal"]==0) & (x_categ["asset"]==2)),["localas"]]=1

x_categ.asset=x_categ.asset.replace({0:0,1:1,2:1,3:2})

#
x_categ1=x_categ[["GM","group_Level_g","LOCAL_RESCONDITION_G","nonlocal","asset"]]

all_iv_detail1 = binn.class_iv(x_categ1,y)

"""---------------------------2.1连续型变量处理---------------------------------"""
#droplist=["fund_month","selfquery_cardquery_in6m","near_open_loan","insurquery_com_num","YEAR_INCOME","com_cardquery_max_in3m"]

"2.1.1:简单输出IV"
iv_detail = binn.cal_iv(pd.concat((x_num,y),axis=1),group=5)

iv_detail.to_csv(r"C:\Users\ts_data\Desktop\iv.csv")

select_feat= iv_detail[iv_detail["ori_IV"]>0.02]["var_name"].drop_duplicates()
X=x[select_feat]
# 连续变量处理方法1 连续变量的相关性

corr1=X.corr()
cul1={}
b=0
for i in corr1.index:
    a=[]
    rowdata = corr1.ix[i,:]

    print (i)
    if any(np.abs(rowdata) >= 0.5):#绝对值有一项>0.75，则运行
        print (i)
        b = b + 1
        cul1[b]=list(rowdata[np.abs(rowdata)>0.6].index)
        corr1=corr1.drop(rowdata[np.abs(rowdata)>0.6].index,axis=1)
        
for a in range(1,b+1):
    iv_detail1=pd.DataFrame([])
    for i in cul1[a]:
        aa=iv_detail[iv_detail["var_name"]==i]
        iv_detail1=pd.concat([iv_detail1,aa],axis=0)
        c=r"C:\Users\ts_data\Desktop"+"\\"+str(a)+".xlsx"
        iv_detail1.to_excel(c)
 


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

X_fs=x[acorr.index]
iv_detail1=pd.DataFrame([])
for i in acorr.index:
    
    a=iv_detail[iv_detail["var_name"]==i]
    iv_detail1=pd.concat([iv_detail1,a],axis=0)
iv_detail1.to_csv(r"C:\Users\ts_data\Desktop\iv1.csv")


'''2.1.2:单调性'''
column_names =x.columns
WOE_detail= pd.DataFrame([])
#排好序的客户特征
X_R= pd.DataFrame([])

for i in column_names:
    print (i)
    X_R_1,WOE_detail_1=binn.mono_bin(data.target, x[i],i)
    WOE_detail=pd.concat([WOE_detail,WOE_detail_1],axis=0)
    X_R=pd.concat([X_R,X_R_1],axis=1)
#    排序
WOE_detail = WOE_detail.sort_values(by=['ori_IV','var_name','max'],ascending=[False,True,True])
    
#aac=WOE_detail[WOE_detail["ori_IV"]>0.02].drop_duplicates("var_name")
select_feat1= WOE_detail[WOE_detail["ori_IV"]>0.02]["var_name"].drop_duplicates()  
X_s=x[select_feat1]

''''''
ac=X_s.corr()
b=-1
for i in ac.index:
    rowdata = ac.ix[i,:]
    b = b + 1
    if any(np.abs(rowdata[:b]) >= 0.5):#绝对值有一项>0.75，则运行
        ac=ac.drop(i)
        ac=ac.drop(i,axis=1)
        b=b-1
ac.index
WOE_detail1=pd.DataFrame()
X_fs=x[ac.index]

for i in ac.index:
    
    a=WOE_detail[WOE_detail["var_name"]==i]
    WOE_detail1=pd.concat([WOE_detail1,a],axis=0)
WOE_detail1.to_csv(r"C:\Users\ts_data\Desktop\WOE_.csv")


#WOE_detail2 单调性
#WOE_detail.to_csv(r"F:\share\model_development\02_DataSet\02_interim\iv_mono.csv")


"""2.1.3:最优分箱"""
x_drop=x_num.drop(["CHILD_COUNT","com_cardquery_max_in3m","cardquery_in3m_max","insurquery_com_num","MONTHLY_EXPENSE"
                   ,"WORK_YEARS","LOAN_60_PASTDUE_CNT","SELF_QUERY_WEEK_FREQUENCY","CARD_OVER_100PCT"
                   ,"CARD_CREDIT_USED_PERCENT","insurquery_in3m","selfquery5_in1m","selfquery5_inl3m","loquery_in3m_f",
                   "loquery_in6m_f","sum_od_in2y","cardquery_in1m_max","com_loquery_max_in3m"
                   ,"com_insurquery_max_in3m","信用卡使用率","job_manu"
            ],axis=1)
    
WOE_bestsplit=pd.DataFrame()
for i in x_drop.columns:
    print(i)
    best_spilt=binn.binContVar(x_drop[i],y,method=1,mmax=4)
    good_total =y.count()-y.sum()
    bad_total = y.sum()
    best_spilt["rate"] = best_spilt['bad']/(best_spilt['bad']+best_spilt['good'])
   
    best_spilt["WOE"] = np.log(best_spilt['bad']/bad_total*good_total/best_spilt['good'])
    best_spilt['MIV'] = ((best_spilt['bad']/bad_total)-(best_spilt['good']/good_total))*best_spilt['WOE']
    best_spilt['ori_IV'] = best_spilt['MIV'].sum()
    best_spilt['var_name'] =i
    WOE_bestsplit=pd.concat([WOE_bestsplit,best_spilt],axis=0)
#    排序
WOE_bestsplit = WOE_bestsplit.sort_values(by=['ori_IV','var_name','upper'],ascending=[False,True,True])
WOE_bestsplit.to_csv(r"F:\share\model_development\02_DataSet\02_interim\WOE_bestsplit_5.csv")
    
#aac=WOE_detail[WOE_detail["ori_IV"]>0.02].drop_duplicates("var_name")
select_feat1= list(WOE_bestsplit[WOE_bestsplit["ori_IV"]>0.02]["var_name"].drop_duplicates())
X_s=x[select_feat1]

''''''
ac=X_s.corr()
b=-1
for i in ac.index:
    rowdata = ac.ix[i,:]
    b=b+1
    if any(np.abs(rowdata[:b]) >= 0.5):#绝对值有一项>0.5，则运行
        ac=ac.drop(i)
        ac=ac.drop(i,axis=1)
        b=b-1
ac.index
X_ms=X_s[ac.index]
WOE_bestsplit1=pd.DataFrame()

for i in X_ms.columns:
    
    ac=WOE_bestsplit[WOE_bestsplit["var_name"]==i]
    WOE_bestsplit1=pd.concat([WOE_bestsplit1,ac],axis=0)
WOE_bestsplit1.to_csv(r"F:\share\model_development\02_DataSet\02_interim\WOE_bestsplit_.csv")

'''处理连续型变量,2.1.4:chi2'''
x_drop2=x_num[select_feat1]

x_n=pd.DataFrame()
woe_chi2=pd.DataFrame()
cut_point = {}
for i in x_drop2.columns:
    print(i)
    cutOffPoints = binn.ChiMerge_MaxInterval_Original(data, i, "target", max_interval = 10)
    x_n[i]=x_num[i].map(lambda x: binn.AssignBin(x, cutOffPoints))
    cut_point[i]=cutOffPoints
    cc=pd.concat([x_n,y],axis=1)
    regroup= binn.CalcWOE(cc, i, "y")
    binn.BadRateMonotone(cc, i, "y")
    cutOffPoints.append(cutOffPoints[-1]*100)
    regroup["max"] = cutOffPoints
    regroup["rate"] = regroup["bad"]/regroup["total"]
    woe_chi2 = pd.concat([woe_chi2,regroup],axis=0)

woe_chi2.to_csv(r"F:\share\model_development\02_DataSet\02_interim\woe_chi2.csv")



''' 变量切分数据导入EXcel后，基于单调性和样本数对分组进行合并。最终结果保存为EXcel'''
'''------------------------------------2.2手动修改IV'分布- 并作为最终输出---------------------------'''
WOE_bestsplit1 = pd.read_excel(r"C:\Users\ts_data\Desktop\bestsplit.xlsx")
WOE_bestsplit1 = WOE_bestsplit1.sort_values(by=['ori_IV','var_name','max'],ascending=[False,True,True])

select_feat1= WOE_bestsplit1["var_name"].drop_duplicates()  
X_s=x[select_feat1]


X_ms = pd.concat([X_s,x_categ1],axis=1)
WOE_detail1 =pd.concat([WOE_bestsplit1,all_iv_detail1],axis=0)

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

#select_feat1= WOE_detail1[WOE_detail1["ori_IV"]>0.04]["var_name"].drop_duplicates()  
#X_ms=X_ms[select_feat1]

new_x=pd.DataFrame()
for i in X_ms.columns:
    print(i)
    new = binn._applyBinwoe(X_ms[i],WOE_detail1[WOE_detail1["var_name"]==i])
    
    new_x=pd.concat([new_x,new],axis=1)

X_ms=new_x


''' 对woe转换后的变量进行筛选'''
from sklearn.linear_model import RandomizedLasso
from sklearn.linear_model import LassoCV
####随机拉索回归选择与y线性关系的变量(稳定性选择2)  
rla = RandomizedLasso()
rla.fit(X_ms,y)
print(X_ms.columns[rla.get_support()])
X_ms = X_ms[X_ms.columns[rla.get_support()]]
#LassoCV LASSO通常用来为其他方法做特征选择,在其他算法中使用。
lassocv = LassoCV()
lassocv.fit(X_ms,y)
print(X_ms.columns[lassocv.coef_ != 0])
X_ms = X_ms[X_ms.columns[lassocv.coef_ != 0]]

###3、显著性检验部分：
import statsmodels.api as sm
logit = sm.Logit(y,X_ms)
result = logit.fit()
print(result.summary())    

###移除2、VIF(方差膨胀因子-判断是否存在多重共线)大于10的
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import preprocessing 
X_scale = preprocessing.scale(X_ms)
vif = [variance_inflation_factor(X_scale, i) for i in range(X_scale.shape[1])]
variables = pd.DataFrame(X_ms.columns)
variables['vif'] = vif
variables_left =  variables[variables['vif']<=10]
X_ms = X_ms[variables_left[0]]

###森林筛选变量 
from sklearn.ensemble import ExtraTreesClassifier

etc = ExtraTreesClassifier()
etc.fit(X_ms, y)
var_name = pd.DataFrame(X_ms.columns.values,columns=['var_name'])
feature_importances = pd.DataFrame(etc.feature_importances_.T,columns=['feature_importances'])
var_name_importances = pd.concat([var_name,feature_importances],axis=1).sort_values('feature_importances',ascending=False)
X_ms = X_ms[var_name_importances[var_name_importances['feature_importances']>0.002]['var_name']]


####RFECV递归特征消除 反复构建SVM选择变量   
from sklearn.feature_selection import RFECV
rfe_svr = RFECV(SVR(kernel='linear'),cv=3)#cv默认等于3。#参数设置为rbf时，没有coef_或者feature_importances_参数
rfe_svr.fit(X_ms, y)
print(X.columns[rfe_svr.support_])

#向前选择，向后淘汰选择变量
X_ms = pick_variables(X_ms,y,method="bs")





'''====================================测试数据   ==================================================='''
#test 数据

test =pd.read_csv(r"F:\share\model_development\02_DataSet\01_Original\test_sample.csv",encoding="utf-8")
test.LOCAL_RESCONDITION_G=test.LOCAL_RESCONDITION_G.replace({0:0,2:0,3:0,6:2,7:1,1:2,4:0,5:1})
test["YEAR_INCOME"] =(test["MONTHLY_SALARY"]+test["MONTHLY_OTHER_INCOME"])*12

test.comp_type=test.comp_type.replace({0:0,1:1,2:1,3:2,4:0,5:2,6:0,7:0})

test.EDUCATION=test.EDUCATION.replace({0:0,1:0,2:1,3:1,4:1,5:1,6:1})


test["GM"] = 0
test.ix[((test["GENDER1"]==0) & (test["MARRIAGE"]==1)),["GM"]]=0 
test.ix[((test["GENDER1"]==0) & (test["MARRIAGE"]!=1)),["GM"]]=1 
test.ix[((test["GENDER1"]==1) & (test["MARRIAGE"]==0)),["GM"]]=0 

test["localas"] = 0
test.ix[((test["nonlocal"]==1) & (test["asset"]==0)),["localas"]]=1 
test.ix[((test["nonlocal"]==1) & (test["asset"]==2)),["localas"]]=0 
test.ix[((test["nonlocal"]==0) & (test["asset"]==0)),["localas"]]=1
test.ix[((test["nonlocal"]==0) & (test["asset"]==2)),["localas"]]=1

test.asset=test.asset.replace({0:0,1:1,2:1,3:2})

test_x = test.drop("target",axis=1)
test_y = test["target"]
test_x=test[X_ms.columns]
new_x=pd.DataFrame()
for i in X_ms.columns:
    print(i)
    new = binn._applyBinwoe(test_x[i],WOE_detail1[WOE_detail1["var_name"]==i])
    new_x=pd.concat([new_x,new],axis=1)

test_x=new_x


'''====================================测试数据   ==================================================='''



'''====================================模型开发==================================================='''

#clf1=quick_make_model(X_ms,y)
#clf1=LogisticRegression(penalty='l1', solver='liblinear',C=C,class_weight={1:class_weight, 0:1})
clf1 = LogisticRegression(penalty='l2')
clf1.fit(X_ms,y)

model_test.ROC_plt(clf1,X_ms,y)
model_test.KS_plt(clf1,X_ms,y,ksgroup=20)

model_test.ROC_plt(clf1,test_x,test_y)
model_test.KS_plt(clf1,test_x,test_y,ksgroup=20)


y_train_pro = clf1.predict_proba(X_ms)
a,b = model_test.lift_lorenz(y_train_pro[:,1],y)
ks_results, ks_ax=model_test.ks_stats(y_train_pro[:,1],y)
#模型系数
formula=model_test.get_lr_formula(clf1,X_ms)
#评分卡分值
scorecard = make_score.make_scorecard(formula,WOE_detail1)

#训练集数据，模型系数，评分卡导出excel
writer = pd.ExcelWriter(r"F:\share\model_development\02_DataSet\02_interim\model_3.xlsx")
scorecard.to_excel(writer,sheet_name='scorecard')
formula.to_excel(writer,sheet_name='formula')
#重设索引
# =============================================================================
# scorecardt=scorecard.set_index(["var_name","rank"])
# scorecardt["score"].to_dict("records") 
# =============================================================================

aa={}
b={}

for i in scorecard.var_name.drop_duplicates():
    a=scorecard[scorecard["var_name"]==i].set_index("woe").T.to_dict("records")
    aa[i]=a[5]
    
X_fs_score = X_ms.replace(aa)

X_fs_score["score"] =X_fs_score.sum(axis=1)

data_1 = pd.concat([X_fs_score,y],axis=1)
#data_1.to_excel(writer,sheet_name='train_result')
#保存Excel
#writer.save()
#测试数据导入Excel
aa={}
b={}

for i in scorecard.var_name.drop_duplicates():
    a=scorecard[scorecard["var_name"]==i].set_index("woe").T.to_dict("records")
    aa[i]=a[5]
    
test_x_score = test_x.replace(aa)

test_x_score["score"] =test_x_score.sum(axis=1)

data_2 = pd.concat([test_x_score,test_y],axis=1)
data_2.to_excel(r"F:\share\model_development\02_DataSet\02_interim\test_3.xlsx")





'''====================================模型检测==================================================='''

#模型分组情况
train_KS=make_score.score_ks(data_1[["score","target"]],1) 
test_KS=make_score.score_ks(data_2[["score","target"]],1) 

print(train_KS["score_KS"],test_KS["score_KS"])

'''测试集 评分卡结果'''

test_y = test["target"]
test_x=test[X_ms.columns]
new_x=pd.DataFrame()
for i in test_x.columns:
    print(i)
    new = binn._applyBinwoe(test_x[i],scorecard[scorecard["var_name"]==i],"rank")
    new_x=pd.concat([new_x,new],axis=1)

test_iv= binn.class_iv(new_x,test_y)
test_iv.to_excel(r"F:\share\model_development\02_DataSet\02_interim\test_iv.xlsx")









