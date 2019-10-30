# -*- coding: utf-8 -*-
"""
Created on Fri Jun 09 17:13:10 2017

@author: potato
"""
from sklearn.linear_model import LogisticRegression#学习速率(步长)参数，权重的正则化参数 
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,roc_auc_score,auc,roc_curve
from imblearn.over_sampling import SMOTE # 导入SMOTE算法模块
from sklearn.model_selection import learning_curve
from sklearn.cross_validation import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV


def ROC_plt(model,X,y):   
        predicted1 = model.predict(X) # 通过分类器产生预测结果  
    
        print("Test set accuracy score: {:.5f}".format(accuracy_score(predicted1, y,)))   
       
        print(classification_report(y, predicted1)) 
        predictions_prob_forest = model.predict_proba(X)
        ''' the different between auc  and  roc_auc_score'''
        false_positive_rate,recall,threshold = roc_curve(y,predictions_prob_forest[:,1])
        roc_auc = auc(false_positive_rate,recall)        
        roc_auc1 = roc_auc_score(y, predicted1)
        print("Area under the ROC curve : %f" % roc_auc)
    
        print("Area under the ROC curve : %f" % roc_auc1)    
            #混淆矩阵 
        m = confusion_matrix(y, predicted1) 
        m   
        plt.figure(figsize=(5,3))
        sns.heatmap(m,annot=True,fmt="d") # 混淆矩阵可视化    
        plt.show()
    
        
    #ROC 可视化
        plt.plot(false_positive_rate,recall)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.show()
        
        return roc_auc
def __tool_sas_rank1(tmp_frame,group):
        '''
        这个按照 sas 公式实现rank分组功能，公式为
        floor(rank*k/(n+1))
        '''
        lenth = len(tmp_frame)
        tmp_frame['rank'] = tmp_frame.ix[:,1].rank(method='min')
        tmp_frame['group_num'] = tmp_frame.apply(lambda row : np.floor(row['rank']*group/(lenth+1)), axis=1)     
    #     ks   
def KS_plt(model,X,y,ksgroup=20):
        predictions_prob = pd.DataFrame(model.predict_proba(X))
        predictions_prob['y'] = y.get_values()
        __tool_sas_rank1(predictions_prob,ksgroup)                
        closPred1 = predictions_prob.groupby('group_num')[1].agg({'minPred1':min,'maxPred1':max})
        colsy = predictions_prob.groupby('group_num')['y'].agg({'bad':sum,'N':len})
        colsy['good'] = colsy['N']-colsy['bad']
        colscumy = colsy.cumsum(0) 
        colscumy = colscumy.rename(columns={'bad':'cum1','N': 'cumN','good':'cum0'}) 
        colscumy['cum1Percent'] = colscumy['cum1']/colscumy['cum1'].max()
        colscumy['cum0Percent'] = colscumy['cum0']/colscumy['cum0'].max() 
        colscumy['cumDiff'] = abs(colscumy['cum1Percent']-colscumy['cum0Percent'])
        ks_file = pd.concat([closPred1,colsy,colscumy],axis=1)
        ks_file['group'] = ks_file.index
        x = np.arange(1,ks_file.shape[0]+1)
        
        plt.plot(x,ks_file['cum0Percent'], label='cum0Percent',marker='o')
        plt.plot(x,ks_file['cum1Percent'], label='cum1Percent',marker='o')
        plt.plot(x,ks_file['cumDiff'], label='cumDiff',marker='o')

        plt.legend()
        plt.title('KS')
        plt.legend(loc='upper left')
        datadotxy=tuple(zip((x+0.2),ks_file['cumDiff']))
        for dotxy in datadotxy:
            plt.annotate(str(round(dotxy[1],2)),xy=dotxy)
            plt.xlabel(u"group", fontproperties='SimHei')

        #plt.savefig("C:\\Users\\123\\Desktop\\KS.png",dpi=2000)
        plt.show()
        

        
        p = pd.DataFrame(model.predict_proba(X),index = X.index)
        p['y'] = y
        proba_y0 = np.array(p[p['y']==0][1])
        proba_y1 = np.array(p[p['y']==1][1])
        ks = stats.ks_2samp(proba_y0,proba_y1)[0]   
        print("K-S score %s"%str(ks))    



def plot_learning_curve(estimator,title,X,y,ylim=None,cv=None,n_jobs=1,
                        train_sizes=np.linspace(.05,1.,20),verbose=0,plot=True):

    train_sizes, train_scores, test_scores = learning_curve( estimator,
                                                            X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)
    train_scores_mean= np.mean(train_scores,axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std =  np.std(test_scores, axis=1)  
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
    return midpoint,diff
    



def get_lr_formula(model,X):
    '''返回回归系数和截距'''
    intercept = pd.DataFrame(model.intercept_) #截距
    coef = model.coef_.T   #模型(回归)系数(相关系数)
    coef = pd.DataFrame(coef)   
    formula = pd.concat([intercept,coef])
    index = ['Intercept']
    index = index + list(X.columns)
    formula.index = index
    formula.reset_index(inplace=True)
    formula.columns = [u'参数',u'估计值']
    return formula
   
"""
提升图和洛伦茨曲线
"""
def lift_lorenz(prob_y, y, k=10):
    """
    plot lift_lorenz curve 
    ----------------------------------
    Params
    prob_y: prediction of model
    y: real data(testing sets)
    k: Section number 
    ----------------------------------
    lift_ax: lift chart
    lorenz_ax: lorenz curve
    """
    # 检查y是否是二元变量
    y_type = type_of_target(y)
    if y_type not in ['binary']:
        raise ValueError('y必须是二元变量')
    # 合并y与y_hat,并按prob_y对数据进行降序排列
    datasets = pd.concat([y, pd.Series(prob_y, name='prob_y', index=y.index)], axis=1)
    datasets.columns = ["y", "prob_y"]
    datasets = datasets.sort_values(by="prob_y", axis=0, ascending=False)
    # 计算正案例数和行数,以及等分子集的行数n
    P = sum(y)
    Nrows = datasets.shape[0]
    n = float(Nrows)/k
    # 重建索引，并将数据划分为子集，并计算每个子集的正例数和负例数
    datasets.index = np.arange(Nrows)
    lift_df = pd.DataFrame()
    rlt = {
            "tile":str(0),
            "Ptot":0,
          }
    lift_df = lift_df.append(pd.Series(rlt), ignore_index=True)
    for i in range(k):
        lo = i*n
        up = (i+1)*n
        tile = datasets.ix[lo:(up-1), :]
        Ptot = sum(tile['y'])
        rlt = {
                "tile":str(i+1),
                "Ptot":Ptot,
                }
        lift_df = lift_df.append(pd.Series(rlt), ignore_index=True)
    # 计算正例比例&累积正例比例
    lift_df['PerP'] = lift_df['Ptot']/P
    lift_df['PerP_cum'] = lift_df['PerP'].cumsum()
    # 计算随机正例数、正例率以及累积随机正例率
    lift_df['randP'] = float(P)/k
    lift_df['PerRandP'] = lift_df['randP']/P
    lift_df.ix[0,:]=0
    lift_df['PerRandP_cum'] = lift_df['PerRandP'].cumsum()
    lift_ax = lift_Chart(lift_df, k)
    lorenz_ax = lorenz_cruve(lift_df)
    return lift_ax, lorenz_ax


def lift_Chart(df, k):
    """
    middle function for lift_lorenz, plot lift Chart
    """
    #绘图变量
    PerP = df['PerP'][1:]
    PerRandP = df['PerRandP'][1:]
    #绘图参数
    fig, ax = plt.subplots()
    index = np.arange(k+1)[1:]
    bar_width = 0.35
    opacity = 0.4
    error_config = {'ecolor': '0.3'}
    rects1 = plt.bar(index, PerP, bar_width,
                 alpha=opacity,
                 color='b',
                 error_kw=error_config,
                 label='Per_p')#正例比例
    rects2 = plt.bar(index + bar_width, PerRandP, bar_width,
                 alpha=opacity,
                 color='r',
                 error_kw=error_config,
                 label='random_P')#随机比例
    plt.xlabel('Group')
    plt.ylabel('Percent')
    plt.title('lift_Chart')
    plt.xticks(index + bar_width / 2, tuple(index))
    plt.legend()
    plt.tight_layout()
    plt.show()

def lorenz_cruve(df):
    """
    middle function for lift_lorenz, plot lorenz cruve
    """
    #准备绘图所需变量
    PerP_cum = df['PerP_cum']
    PerRandP_cum = df['PerRandP_cum']
    decilies = df['tile']
    #绘制洛伦茨曲线
    plt.plot(decilies, PerP_cum, 'm-^', label='lorenz_cruve')#lorenz曲线
    plt.plot(decilies, PerRandP_cum, 'k-.', label='random')#随机
    plt.legend()
    plt.xlabel("decilis")#等份子集
    plt.title("lorenz_cruve", fontsize=10)#洛伦茨曲线
    plt.show()  

    