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
from scorecard import iv,bining,model_test

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
data1 = data.drop("target",axis=1)




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
object_iv = iv.class_iv(data1[["addr_state"]],y)
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


'''
连续变量
'''
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
All_f_iv=pd.DataFrame()
for i in cul1:
    print (cul1[i])
    cul_woe= iv.cal_iv(pd.concat([data1[cul1[i]],y],axis=1),group=5)
    best_ = list(cul_woe["var_name"].drop_duplicates())[0]
    cul_iv=pd.concat([cul_iv,cul_woe[cul_woe["var_name"]==best_]],axis=0)
    All_f_iv = pd.concat([All_f_iv,cul_woe],axis=0)

num_iv_2=cul_iv

#连续变量处理方法2
num_iv = iv.cal_iv(pd.concat([data1[numColumns],y],axis=1),group=5)

acc=pd.DataFrame()

acc["open_il_24m"] = data1["inq_last_12m"]-data1["inq_last_6mths"]
acc["open_il_24m"]=acc["open_il_24m"].fillna(-2)
acc.ix[acc["open_il_24m"]==np.inf ,"open_il_24m"] =-1
acc_iv = iv.cal_iv(pd.concat([acc["open_il_24m"],y],axis=1),group=10)


num_iv_2 = num_iv[num_iv["ori_IV"]>0.02]

num_sel_feat= num_iv[num_iv["ori_IV"]>0.02]["var_name"].drop_duplicates()
x_num=data1[num_sel_feat]

data1.to_csv(r"F:\python\python\Credit\LoanStats_2016Q1\train_two.csv")
object_iv.to_csv(r"F:\python\python\Credit\LoanStats_2016Q1\object_iv.csv")
num_iv_2.to_csv(r"F:\python\python\Credit\LoanStats_2016Q1\num_iv_2.csv")

# =============================================================================
# #
# num_iv_best=pd.DataFrame()
# for i in numColumns:
#     print(i)
#     best_spilt=bining.binContVar(data1[i],y,method=1,mmax=5)
#     good_total =y.count()-y.sum()
#     bad_total = y.sum()
#     best_spilt["rate"] = best_spilt['bad']/(best_spilt['bad']+best_spilt['good'])
#    
#     best_spilt["WOE"] = np.log(best_spilt['bad']/bad_total*good_total/best_spilt['good'])
#     best_spilt['MIV'] = ((best_spilt['bad']/bad_total)-(best_spilt['good']/good_total))*best_spilt['WOE']
#     best_spilt['ori_IV'] = best_spilt['MIV'].sum()
#     best_spilt['var_name'] =i
#     num_iv_best=pd.concat([num_iv_best,best_spilt],axis=0)
# #    排序
# num_iv_best = num_iv_best.sort_values(by=['ori_IV','var_name','upper'],ascending=[False,True,True])
# 
# 
# =============================================================================

data1=pd.read_csv(r"F:\python\python\Credit\LoanStats_2016Q1\train_two.csv",encoding="gbk",index_col =0,low_memory=False)
object_iv=pd.read_csv(r"F:\python\python\Credit\LoanStats_2016Q1\object_iv.csv",encoding="gbk",index_col =0,low_memory=False)
num_iv_2=pd.read_csv(r"F:\python\python\Credit\LoanStats_2016Q1\num_iv_2.csv",encoding="gbk",index_col =0,low_memory=False)

data1["home_ownership"]=data1["home_ownership"].replace({"ANY":"MORTGAGE"})
data1["grade"]=data1["grade"].replace({"E":"E","F":"E","G":"E"})



'''==============================2.3:合并数据,woe编码================='''

woe =pd.concat([num_iv_2,object_iv],axis=0).sort_values(by=['ori_IV','var_name','max'],ascending=[False,True,True])
sel_feat= woe[woe["ori_IV"]>=0.01]["var_name"].drop_duplicates()

X = data1[sel_feat]

new_x=pd.DataFrame()
for i in X.columns:
    print(i)
    new = bining._applyBinwoe(X[i],woe[woe["var_name"]==i])
    
    new_x=pd.concat([new_x,new],axis=1)
X_ms=new_x


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

#向前选择，向后淘汰选择变量
X_ms = pick_variables(X_ms,y,method="bs")


###移除2、VIF(方差膨胀因子-判断是否存在多重共线)大于10的
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import preprocessing 
X_scale = preprocessing.scale(X_ms)
vif = [variance_inflation_factor(X_scale, i) for i in range(X_scale.shape[1])]
variables = pd.DataFrame(X_ms.columns)
variables['vif'] = vif
variables_left =  variables[variables['vif']<=10]
X_ms = X_ms[variables_left[0]]

#select_feat1= WOE_detail1[WOE_detail1["ori_IV"]>0.04]["var_name"].drop_duplicates()  
#X_ms=X_ms[select_feat1]

X_ms=X_ms.drop(["all_util","mths_since_recent_inq","bc_open_to_buy","mo_sin_old_rev_tl_op","mo_sin_rcnt_tl"],axis=1)
from sklearn.linear_model import LogisticRegression#学习速率(步长)参数，权重的正则化参数 

clf1 = LogisticRegression(class_weight="balanced")
clf1.fit(X_ms,y)

model_test.ROC_plt(clf1,X_ms,y)
model_test.KS_plt(clf1,X_ms,y,ksgroup=20)
y_train_pro = clf1.predict_proba(X_ms)
a,b = moudle_evaluate.lift_lorenz(y_train_pro[:,1],y)


model_test.ROC_plt(clf1,test_x,test_y)
model_test.KS_plt(clf1,test_x,test_y,ksgroup=20)


formula=get_lr_formula(clf1,X_ms)

scorecard = iv.make_scorecard(formula,woe)


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




train_KS=score_ks(data_1[["score","y"]],1) 
test_KS=score_ks(data_2[["score","y"]],1) 

print(train_KS["score_KS"],test_KS["score_KS"])























