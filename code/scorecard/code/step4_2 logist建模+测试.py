# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 15:00:48 2018

@author: ly
"""

data1=pd.read_csv(r"F:\python\python\Credit\LoanStats_2016Q1\train_two.csv",encoding="gbk",index_col =0,low_memory=False)
object_iv=pd.read_csv(r"F:\python\python\Credit\LoanStats_2016Q1\object_iv.csv",encoding="gbk",index_col =0,low_memory=False)
num_iv_2=pd.read_csv(r"F:\python\python\Credit\LoanStats_2016Q1\num_iv_2.csv",encoding="gbk",index_col =0,low_memory=False)

data1["home_ownership"]=data1["home_ownership"].replace({"ANY":"MORTGAGE"})
data1["grade"]=data1["grade"].replace({"E":"E","F":"E","G":"E"})



'''==============================2.3:合并数据,woe编码================='''

woe =pd.concat([num_iv_2,object_iv],axis=0).sort_values(by=['ori_IV','var_name','max'],ascending=[False,True,True])
sel_feat= woe[woe["ori_IV"]>=0.01]["var_name"].drop_duplicates()

X = data1[sel_feat].drop("mob",axis=1)

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
    if any(np.abs(rowdata[:b]) >= 0.6):#绝对值有一项>0.75，则运行
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

X_ms=X_ms.drop(["all_util","grade"],axis=1)
from sklearn.linear_model import LogisticRegression#学习速率(步长)参数，权重的正则化参数 

clf1 = LogisticRegression(class_weight="balanced")
clf1.fit(X_ms,y)

model_test.ROC_plt(clf1,X_ms,y)
model_test.KS_plt(clf1,X_ms,y,ksgroup=20)

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
