# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 16:45:52 2017

@author: potato

#用于辅助建模，综合各种工具
"""

import pandas as pd
import numpy as np

from scorecard.binn import *

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score,learning_curve
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.linear_model import RandomizedLasso
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression#学习速率(步长)参数，权重的正则化参数 
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
from scorecard.model_evaluation_plot import *
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier


 
    
def quick_make_model(x,y,model="LR",best_params=None):
    #快速建模方法
    #best_params 就是之前最優化時獲取的结果
    if model == "LR":
        if best_params == None:
            model = LogisticRegression()
        else:
            model = best_params['lr'] 
        model.fit(x, y)
        model_evaluation_plot(model,x,y)
        
        return model


def get_lr_formula(model,X):
    intercept = pd.DataFrame(model.intercept_)
    coef = model.coef_.T   #model.coef_ 模型的参数
    coef = pd.DataFrame(coef)
    formula = pd.concat([intercept,coef])
    index = ['Intercept']
    index = index + list(X.columns)
    formula.index = index
    formula.reset_index(inplace=True)
    formula.columns = [u'参数',u'估计值']
    return formula
        


def pick_variables(x,y,descover=True,method="rlr",threshold=0.25,sls=0.05):#默认阈值0.25
    #挑选变量助手
    if method == "rlr":
        #随机逻辑回归选择与y线性关系的变量(稳定性选择1)。
        #在不同数据子集和特征子集上运行特征选择算法(rlr)，最终汇总选择结果
        rlr = RandomizedLogisticRegression(selection_threshold=threshold)
        rlr.fit(x,y)
        scoretable = pd.DataFrame(rlr.all_scores_,index = x.columns,columns = ['var_score'])
        columns_need = list(x.columns[rlr.get_support()])
        x = x[columns_need]
    if method =="bs":   
        import statsmodels.formula.api as smf#导入相应模块
        data = pd.concat([x, y], axis=1)#合并数据
        #提取X，y变量名
        var_list = x.columns
        response = y.name
        #首先对所有变量进行模型拟合
        while True:
            logit = smf.Logit(y,x)
            mod = logit.fit()
            print(mod.summary())
            p_list = mod.pvalues.sort_values()
            if p_list[-1] > sls:
                #提取p_list中最后一个index
                var = p_list.index[-1]
                #var_list中删除
                x = x.drop(var,axis=1)   
            else:
                break

    
    if method =="fs":          
        data = pd.concat([x, y], axis=1)#合并数据
    #提取X，y变量名
        var_list = x.columns
        response = y.name
        #首先对所有变量进行模型拟合
        while True:
            formula = "{} ~ {} + 1".format(response, ' + '.join(var_list))
            mod = smf.logit(formula, data).fit()
            p_list = mod.pvalues.sort_values()
            if p_list[-1] > sls:
                #提取p_list中最后一个index
                var = p_list.index[-1]
                print(var)
                #var_list中删除
                var_list = var_list.drop(var) 
            else:
                break
        x = x[var_list]

    '''
    注意这里调用的是statsmodels.api里的逻辑回归。这个回归模型可以获取每个变量的显著性p值，p值越大越不显著，当我们发现多于一个变量不显著时，不能一次性剔除所有的不显著变量，因为里面可能存在我们还未发现的多变量的多重共线性，我们需要迭代的每次剔除最不显著的那个变量。 
    上面迭代的终止条件： 
    ①剔除了所有的不显著变量 
    ②剔除了某一个或某几个变量后，剩余的不显著变量变得显著了。（说明之前存在多重共线性）
    '''
    if method =="rfc":   
        RFC = RandomForestClassifier(n_estimators=200,max_depth=5,class_weight="balanced")
        RFC_Model = RFC.fit(x,y)
        features_rfc = x.columns
        featureImportance = {features_rfc[i]:RFC_Model.feature_importances_[i] for i in range(len(features_rfc))}
        featureImportanceSorted = sorted(featureImportance.items(),key=lambda x: x[1], reverse=True)
        # we selecte the top 10 features
        features_selection = [k[0] for k in featureImportanceSorted[:15]]
        
        x = x[features_selection]
        x['intercept'] = [1]*x.shape[0]
        
        LR = sm.Logit(y, x).fit()
        summary = LR.summary()
        print(summary)
        x=x.drop("intercept",axis=1)

    return x

def pick_variables_bylist(x,columns_need):#默认阈值0.25
    #挑选变量助手
    x = x[columns_need]    
    return x               

def model_optimizing(x,y,model="LR"):
    if model == "LR":
        pipline = Pipeline([('lr',LogisticRegression())
                #('sgd',SGDClassifier(loss='log'))#LR
                #('sgd',SGDClassifier(loss='hinge'))#SVM
                #('svm',SVC()) 
                ])
        parameters = {
          #C正则化的系数
          'lr__penalty': ('l1','l2'),'lr__C': (0.01,0.1,10,1),'lr__max_iter':(80,150,100),
#              #随机梯度下降分类器。alpha正则化的系数,n_iter在训练集训练的次数，learning_rate为什么是alpha的倒数
#              'sgd__alpha':(0.00001,0.000001,0.0001),'sgd__penalty':('l1','l2','elasticnet'),'sgd__n_iter':(10,50,5),  
#              #核函数，将数据映射到高维空间中，寻找可区分数据的高维空间的超平面
#              'svm__C':(2.5,1),'svm__kernel':('linear','poly','rbf'),
          }

        grid_search = GridSearchCV(pipline,parameters,n_jobs=6,scoring='recall',cv=3)
        grid_search.fit(x, y) 
        print('Best score: %0.3f' % grid_search.best_score_)
        print('Best parameters set:')
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print('\t%s: %r' % (param_name, best_parameters[param_name]))    
        return best_parameters


#用sklearn的learning_curve得到training_score和cv_score，使用matplotlib画出learning

def plot_learning_curve(X,y,estimator,title="学习曲线",ylim=None,cv=None,n_jobs=1,
                    train_sizes=np.linspace(.05,1.,20),verbose=0,plot=True):
#==============================================================================
# 画出data在某模型上的learning curve.\n",
# 参数解释\n",
# ----------\n",
# estimator : 你用的分类器。\n",
# title : 表格的标题。\n",
# X : 输入的feature，numpy类型\n",
# y : 输入的target vector\n",
# ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点\n",
# cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)\n",
# n_jobs : 并行的的任务数(默认1)\n",    
#==============================================================================
    print("学习曲线")
    train_sizes, train_scores, test_scores = learning_curve( estimator,
                                                            X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)
    train_scores_mean=np.mean(train_scores,axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)  
    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
              plt.ylim(*ylim)
        plt.xlabel("训练样本数")
        plt.ylabel("得分")
        plt.gca().invert_yaxis()
        plt.grid()
     
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 
                            alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 
                            alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label="训练集上得分")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label="交叉验证集上得分")

        plt.legend(loc="best")
           
        plt.draw()
        plt.gca().invert_yaxis()
        plt.show()
    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff








if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    
    iris = load_iris()
    X, y = iris.data, iris.target
    X.shape
    X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
    
    
    
    
    
    
    
    
    