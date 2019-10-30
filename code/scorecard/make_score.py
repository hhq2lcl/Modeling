"""
评分卡
"""

import numpy as np
import statsmodels.api as sm
#import re
import pandas as pd


def tool_group_rank(tmp_frame,group):

    c,s = pd.qcut(tmp_frame.iloc[:,0].unique(),group,retbins =1) 
    def get_group_num(x):
        for i in range(len(s-1)):
            if x<=s[i+1]:
                return i
    tmp_frame['group_num'] = tmp_frame.iloc[:,0].apply(get_group_num)


def make_scorecard(formular,woe,basescore=600.0,base_odds=50.0/1.0,pdo=50.0):
    
#    #step6 生成评分卡
#    basescore = float(600)
#    base_odds = 50.0/1.0
#    pdo = float(50)
        #计算所需要的参数

    a = formular[formular[u"参数"] == "Intercept"].iloc[0,1]
    formular = formular.iloc[1:,:]
    n = float(len(formular))
    factor = pdo/np.log(2)
    offset = basescore - factor*np.log(base_odds)
#保留两位小数
        #生成评分卡
    scorecard = pd.DataFrame()
    for i in formular["参数"]:
        woe_frame = woe[woe['var_name'] == i][['var_name','min','max','woe',"rank","total_rate"]]
        beta_i = formular[formular[u"参数"] == i][u"估计值"].iloc[0]
        woe_frame['score'] = woe_frame['woe'].apply(lambda woe : round(offset/n - factor*(a/n+beta_i*woe)))
        scorecard = pd.concat((scorecard,woe_frame),axis=0)
        
    return scorecard

    
    
def score_ks(data,types=1,group=10,ycol=-1):
    '''计算评分卡 KS'''
    all_iv_detail = pd.DataFrame([])

    
    if type(ycol) == int:
        ycol = data.columns[ycol]
    
    if type(group) == int:
        column_names = data.columns[data.columns != ycol]
    elif isinstance(group,pd.DataFrame):
        column_names = group['var_name'].unique()
    else:
        print("argument 'group' type is wrong")
        return 0,0        

#    flag_ = 0    
    for i in column_names: #默认y在最后一列
        print(i)

        tmp = pd.concat([pd.DataFrame(data[i]),data[[ycol]]],axis=1)#tmp是临时的iv计算数据框
        tmp = tmp.astype('float')
        tmp.sort_values(by=tmp.columns[0],inplace=True)
        if type(types) == 1:
            tool_sas_rank(tmp,group) #使用上面写的分组函数
        else:
            tool_group_rank(tmp,group)
        grouped = tmp.groupby(tmp['group_num'])

        cols = grouped[tmp.columns[0]].agg({'min':min,'max':max})
        cols['group'] = range(len(cols))
        def len_minus_sum(x):
            ''' 默认了 1 代表坏人'''
            return len(x)-sum(x)
        col2 = grouped[tmp.columns[1]].agg({'y1_num':sum,'y0_num':len_minus_sum,'N':'size'})      
        cols = pd.concat([cols,col2],axis=1)
        bad_totl_num = float(tmp[tmp.columns[1]].sum())
        good_totl_num = float(len(tmp) - bad_totl_num)
        cols['bad_cum'] = cols['y1_num'].cumsum()/bad_totl_num
        cols['good_cum'] = cols['y0_num'].cumsum()/good_totl_num
        cols['y1_percent'] = cols['y1_num'] / bad_totl_num
        cols['y0_percent'] = cols['y0_num'] /good_totl_num
        cols["y0/y1"] =cols["y0_percent"]/cols["y1_percent"]
        cols["od"]=cols["y1_num"]/cols["N"]
        cols['total_percent'] = cols['N'] / (bad_totl_num+ good_totl_num)
        cols['woe'] = np.log(cols['y0_percent']/cols['y1_percent'])
        cols.ix[cols['woe'] == np.inf,'woe'] = 0 # 分母为0的先设置为0吧
        cols.ix[cols['woe'] == -np.inf,'woe'] = 0 # 分母为0的先设置为0吧
        cols['MIV'] = (cols['y0_percent']-cols['y1_percent'])*cols['woe']
        cols['ori_IV'] = cols['MIV'].sum()
        cols['KS'] =  cols['bad_cum']-cols['good_cum']
        cols['score_KS'] =  cols['KS'].max()
       
        cols['var_name'] = i

        all_iv_detail = pd.concat([all_iv_detail,cols],axis=0)
#        flag_ = flag_+1
#        if flag_>3:
#            break
   
    all_iv_detail = all_iv_detail.sort_values(by=['ori_IV','var_name','max'],ascending=[False,True,True])
    return all_iv_detail  



