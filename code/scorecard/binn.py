# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 17:30:40 2017
几种分箱方法
等距
等频
卡方
@author: ly
"""
import pandas as pd
import numpy as np
import scipy.stats.stats as stats


#名义变量切分
def class_iv(x,y):
    all_iv_detail = pd.DataFrame([])
    bad_sample = y.sum()
    good_sample = y.count()-y.sum()
    #    flag_ = 0    
    for i in x.columns: #默认y在最后一列
        print(i)
        tmp = pd.concat([x[i],y],axis=1)#tmp是临时的iv计算数据框
        tmp.sort_values(by=tmp.columns[0],inplace=True)
    
        grouped = tmp.groupby(tmp[i])
        cols=pd.DataFrame()
        cols['max'] =  grouped[i].max()
        cols['min'] =  grouped[i].min()
        cols['rank'] = range(len(grouped))
        cols['bad'] = grouped.sum()
        cols['good'] = grouped.count()-grouped.sum()
        cols['BAD_rate'] = cols['bad'] / bad_sample
        cols['GOOD_rate'] = cols['good'] / good_sample
        cols['total'] = ( cols['good']+ cols['bad'])
        cols["total_rate"]=cols['total']/y.count()
        cols["rate"]=grouped.mean()
        cols['woe'] = np.log(cols['BAD_rate']/cols['GOOD_rate'])
        cols.ix[cols['woe'] == np.inf,'woe'] = 0 # 分母为0的先设置为0吧
        cols.ix[cols['woe'] == -np.inf,'woe'] = 0 # 分母为0的先设置为0吧
        cols['MIV'] = (cols['BAD_rate']-cols['GOOD_rate'])*cols['woe']
        cols['ori_IV'] = cols['MIV'].sum()
        cols['var_name'] = i
    
        all_iv_detail = pd.concat([all_iv_detail,cols],axis=0)
    
    all_iv_detail = all_iv_detail.sort_values(by=['ori_IV','var_name'],ascending=[False,True])
    
    return all_iv_detail




''' 等距切分'''
def tool_sas_rank(tmp_frame,group):

    lenth = len(tmp_frame)
    tmp_frame['rank'] = tmp_frame.ix[:,0].rank(method='min')
    tmp_frame['group_num'] = tmp_frame.apply(lambda row : np.floor(row['rank']*group/(lenth+1)), axis=1)    
    
def tool_group_bygiven(tmp_frame,group):
    s = group['max']
    s.reset_index(drop=True,inplace=True)
    def get_group_num(x):
        for i in range(len(s)):
            if x<=s[i]:
                return i
    tmp_frame['group_num'] = tmp_frame.iloc[:,0].apply(get_group_num)

def cal_iv(data,group=20,ycol=-1):

    '''简单的等分数据拆分'''
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
   
    for i in column_names: 
        print(i)

        tmp = pd.concat([pd.DataFrame(data[i]),data[[ycol]]],axis=1)
        tmp = tmp.astype('float')
        tmp.sort_values(by=tmp.columns[0],inplace=True)
        if type(group) == int:
            tool_sas_rank(tmp,group) 
        else:
            tool_group_bygiven(tmp,group[group['var_name']==i])
        grouped = tmp.groupby(tmp['group_num'])

        cols = grouped[tmp.columns[0]].agg({'min':min,'max':max})
        cols['var_name'] = i

        cols['rank'] = range(len(cols))
        def len_minus_sum(x):
            return len(x)-sum(x)
        col2 = grouped[tmp.columns[1]].agg({'bad':sum,'good':len_minus_sum,'total':'size'})      
        cols = pd.concat([cols,col2],axis=1)
        bad_totl_num = float(tmp[tmp.columns[1]].sum())
        good_totl_num = float(len(tmp) - bad_totl_num)
        cols["rate"]=cols["bad"]/cols["total"]
        
        cols['BAD_rate'] = cols['bad'] / bad_totl_num
        cols['GOOD_rate'] = cols['good'] / good_totl_num
        cols['total_rate'] = cols['total'] / (bad_totl_num + good_totl_num)
        cols['woe'] = np.log(cols['BAD_rate']/cols['GOOD_rate'])
        cols.ix[cols['woe'] == np.inf,'woe'] = 0 # 分母为0的先设置为0吧
        cols.ix[cols['woe'] == -np.inf,'woe'] = 0 # 分母为0的先设置为0吧
        cols['MIV'] = (cols['BAD_rate']-cols['GOOD_rate'])*cols['woe']
        cols['ori_IV'] = cols['MIV'].sum()

        all_iv_detail = pd.concat([all_iv_detail,cols],axis=0)
#        flag_ = flag_+1
#        if flag_>3:
#            break
   
    all_iv_detail = all_iv_detail.sort_values(by=['ori_IV','var_name','max'],ascending=[False,True,True])
    return all_iv_detail  


'''等距切分+单调性'''
def mono_bin(Y, X, i,n=5):
  # fill missings with median
    X_R_1= pd.DataFrame([])
    X2 = X.fillna(np.median(X))
    sample = len(X)
    bad_sample=Y.sum()
    good_sample=sample-bad_sample
    r = 0

    while np.abs(r) < 1 :
        
        d1 = pd.DataFrame({"X": X2, "Y": Y})
        tool_sas_rank(d1,n) 
        d2 = d1.groupby('group_num', as_index = True)
        '''斯皮尔曼相关性是两个数据集之间关系单调性的非参数度量。与Pearson相关不同，
        Spearman相关并没有假设两个数据集都是正常分布的。和其他相关系数一样，这个值在- 1和+ 1之间变化，0表示没有相关性。
        - 1或+ 1的相关性意味着一种完全单调的关系。正相关意味着x增加，y也增加，负相关意味着x增加y减小。
        p值大致表示一个不相关的系统产生数据集的概率，这些数据集至少与从这些数据集计算的数据集一样极端。p值并不完全可靠，
        但对于大于500个左右的数据集来说可能是合理的。'''
        r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
        n = n - 1
    X_R_1[i]=d1['group_num'].rank(method='dense')-1

    d3 = pd.DataFrame(d2.min().X.values, columns = ['min'])
    d3["rank"] = range(len(d2))
    d3['max' ] = d2.max().X.values
    d3["bad"] = d2.sum().Y.values
    d3["good"] = d2.count().Y.values-d2.sum().Y.values
    d3["BAD_rate"] = d2.sum().Y.values/bad_sample
    d3["GOOD_rate"] = (d2.Y.count().values-d2.sum().Y.values)/good_sample
    d3['total'] = d2.count().Y.values
    d3['rate'] = d2.mean().Y.values
    d3["woe"] = np.log(d3["BAD_rate"]/ d3["GOOD_rate"])
    d4 = (d3.sort_values(by = 'min')).reset_index(drop = True)
    d4['MIV'] = (d4['BAD_rate']-d4['GOOD_rate'])*d4['woe']
    d4['ori_IV'] = d4['MIV'].sum()
    d4['var_name'] = i
  
    print ("=" * 60)
    print (d4)

    return X_R_1,d4



#卡方分箱 按照最大区间数进行分箱代码
def Chi2(df, total_col, bad_col, overallRate):
    '''
    :param df: the dataset containing the total count and bad count
    :param total_col: total count of each value in the variable
    :param bad_col: bad count of each value in the variable
    :param overallRate: the overall bad rate of the training set
    :return: the chi-square value
    '''
    df2 = df.copy()
    df2['expected'] = df[total_col].apply(lambda x: x*overallRate)
    combined = zip(df2['expected'], df2[bad_col])
    chi = [(i[0]-i[1])**2/i[0] for i in combined]
    chi2 = sum(chi)
    return chi2


### ChiMerge_MaxInterval: split the continuous variable using Chi-square value by specifying the max number of intervals
def ChiMerge_MaxInterval_Original(df, col, target, max_interval = 5):
    '''
    :param df: the dataframe containing splitted column, and target column with 1-0
    :param col: splitted column
    :param target: target column with 1-0
    :param max_interval: the maximum number of intervals. If the raw column has attributes less than this parameter, the function will not work
    :return: the combined bins
    '''
    colLevels = set(df[col])
    # since we always combined the neighbours of intervals, we need to sort the attributes
    colLevels =sorted(list(colLevels))
# 先对这列数据进行排序，然后在计算分箱
    N_distinct = len(colLevels)
    if N_distinct <= max_interval:  #If the raw column has attributes less than this parameter, the function will not work
        print ("The number of original levels for {} is less than or equal to max intervals".format(col))
        return colLevels[:-1]
    else:
        #Step 1: group the dataset by col and work out the total count & bad count in each level of the raw column
        total = df.groupby([col])[target].count()
        total = pd.DataFrame({'total':total})
        bad = df.groupby([col])[target].sum()
        bad = pd.DataFrame({'bad':bad})
        regroup =  total.merge(bad,left_index=True,right_index=True, how='left')##将左侧，右侧的索引用作其连接键。
        regroup.reset_index(level=0, inplace=True)
        N = sum(regroup['total'])
        B = sum(regroup['bad'])
        #the overall bad rate will be used in calculating expected bad count
        overallRate = (B*1.0 /N)
        #　统计坏样本率
        # initially, each single attribute forms a single interval
        groupIntervals = [[i] for i in colLevels]
        ## 类似于[[1],[2],[3,4]]其中每个[.]为一箱
        groupNum = len(groupIntervals)
        while(len(groupIntervals)>max_interval): 
            #the termination condition: the number of intervals is equal to the pre-specified threshold
            # in each step of iteration, we calcualte the chi-square value of each atttribute
            chisqList = []
            for interval in groupIntervals:
                df2 = regroup.loc[regroup[col].isin(interval)]
                chisq = Chi2(df2, 'total','bad',overallRate)
                chisqList.append(chisq)
            #find the interval corresponding to minimum chi-square, and combine with the neighbore with smaller chi-square
            min_position = chisqList.index(min(chisqList))
            if min_position == 0:
                ## 如果最小位置为0,则要与其结合的位置为１
                combinedPosition = 1
            elif min_position == groupNum - 1:
                combinedPosition = min_position -1
            else:
                ## 如果在中间，则选择左右两边卡方值较小的与其结合
                if chisqList[min_position - 1]<=chisqList[min_position + 1]:
                    combinedPosition = min_position - 1
                else:
                    combinedPosition = min_position + 1
            groupIntervals[min_position] = groupIntervals[min_position]+groupIntervals[combinedPosition]
            # after combining two intervals, we need to remove one of them
            groupIntervals.remove(groupIntervals[combinedPosition])
            groupNum = len(groupIntervals)
        groupIntervals = [sorted(i) for i in groupIntervals]
        ## 对每组的数据安从小到大排序
        cutOffPoints = [i[-1] for i in groupIntervals[:-1]]
        ## 提取出每组的最大值，也就是分割点
        return cutOffPoints
#以卡方阈值作为终止分箱条件：
def ChiMerge_MinChisq(df, col, target, confidenceVal = 3.841):
    '''
    :param df: the dataframe containing splitted column, and target column with 1-0
    :param col: splitted column
    :param target: target column with 1-0
    :param confidenceVal: the specified chi-square thresold, by default the degree of freedom is 1 and using confidence level as 0.95
    :return: the splitted bins
    '''
    colLevels = set(df[col])
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total':total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad':bad})
    regroup =  total.merge(bad,left_index=True,right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    N = sum(regroup['total'])
    B = sum(regroup['bad'])
    overallRate = B*1.0/N
    colLevels =sorted(list(colLevels))
    groupIntervals = [[i] for i in colLevels]
    groupNum  = len(groupIntervals)
    while(1):   #the termination condition: all the attributes form a single interval; or all the chi-square is above the threshould
        if len(groupIntervals) == 1:
            break
        chisqList = []
        for interval in groupIntervals:
            df2 = regroup.loc[regroup[col].isin(interval)]
            chisq = Chi2(df2, 'total','bad',overallRate)
            chisqList.append(chisq)
        min_position = chisqList.index(min(chisqList))
        if min(chisqList) >=confidenceVal:
            break
        if min_position == 0:
            combinedPosition = 1
        elif min_position == groupNum - 1:
            combinedPosition = min_position -1
        else:
            if chisqList[min_position - 1]<=chisqList[min_position + 1]:
                combinedPosition = min_position - 1
            else:
                combinedPosition = min_position + 1
        groupIntervals[min_position] = groupIntervals[min_position]+groupIntervals[combinedPosition]
        groupIntervals.remove(groupIntervals[combinedPosition])
        groupNum = len(groupIntervals)
    return groupIntervals

def CalcWOE(df, col, target):
    '''
    :param df: dataframe containing feature and target
    :param col: 注意col这列已经经过分箱了，现在计算每箱的WOE和总的IV。
    :param target: good/bad indicator
    :return: 返回每箱的WOE(字典类型）和总的IV之和。
    '''
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    N = sum(regroup['total'])
    B = sum(regroup['bad'])
    regroup['good'] = regroup['total'] - regroup['bad']
    G = N - B
    regroup['bad_pcnt'] = regroup['bad'].map(lambda x: x*1.0/B)
    regroup['good_pcnt'] = regroup['good'].map(lambda x: x * 1.0 / G)
    regroup['WOE'] = regroup.apply(lambda x: np.log(x.good_pcnt*1.0/x.bad_pcnt),axis = 1)
    regroup['IV']= regroup.apply(lambda x: (x.good_pcnt-x.bad_pcnt)*np.log(x.good_pcnt*1.0/x.bad_pcnt),axis = 1)
    regroup["IV_sum"] = regroup['IV'].sum()
    regroup["var_name"]=col
    regroup.rename(columns={col:'rank'}, inplace = True)
    return regroup

'''
分箱的注意点

对于连续型变量做法:

使用ChiMerge进行分箱
如果有特殊值，把特殊值单独分为一组，例如把-1单独分为一箱。
计算这个连续型变量的每个值属于那个箱子，得出箱子编号。以所属箱子编号代替原始值。
'''
def AssignBin(x, cutOffPoints):
    '''
    :param x: the value of variable
    :param cutOffPoints: the ChiMerge result for continous variable
    :return: bin number, indexing from 0
    for example, if cutOffPoints = [10,20,30], if x = 7, return Bin 0. If x = 35, return Bin 3
    '''
    numBin = len(cutOffPoints) + 1
    if x<=cutOffPoints[0]:
        return 0
    elif x > cutOffPoints[-1]:
        return numBin-1
    else:
        for i in range(0,numBin-1):
            if cutOffPoints[i] < x <=  cutOffPoints[i+1]:
                return i+1

'''检查分箱以后每箱的bad_rate的单调性，如果不满足，那么继续进行相邻的两箱合并，知道bad_rate单调为止。(可以放宽到U型)'''
## determine whether the bad rate is monotone along the sortByVar
def BadRateMonotone(df, sortByVar, target):
    # df[sortByVar]这列数据已经经过分箱
    df2 = df
    total = df2.groupby([sortByVar])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df2.groupby([sortByVar])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    combined = zip(regroup['total'],regroup['bad'])
    badRate = [x[1]*1.0/x[0] for x in combined]
    badRateMonotone = [badRate[i]<badRate[i+1] for i in range(len(badRate)-1)]
    Monotone = len(set(badRateMonotone))
    if Monotone == 1:
        return True
    else:
        return False


'''最优切分+降基'''
from sklearn.utils.multiclass import type_of_target
#type_of_target可以检查变量类型，连续或二分类等

def check_target_binary(y):
    """
    check if the target variable is binary(二元的)
    ------------------------------
    Param(参数)
    y:exog variable,pandas Series contains binary variable
    ------------------------------
    Return
    if y is not binary, raise a error   
    """
    y_type = type_of_target(y)
    if y_type not in ['binary']:
        raise ValueError('目标变量必须是二元的！')


def isNullZero(x):
    """
    check x is null or equal zero
    -----------------------------
    Params
    x: data 
    -----------------------------
    Return
    bool obj
    """
    cond1 = np.isnan(x)
    cond2 = x==0
    return cond1 or cond2


def Gvalue(binDS, method):
    """
    Calculation of the metric of current split
    ----------------------------------------
    Params
    binDS: pandas dataframe
    method: int obj, metric to split x(1:Gini, 2:Entropy, 3:person chisq, 4:Info value)
    -----------------------------------------
    Return
    M_value: float or np.nan
    """    
    R = binDS['bin'].max()
    N = binDS['total'].sum()
    
    N_mat = np.empty((R,3))
    # calculate sum of 0,1
    N_s = [binDS[0].sum(), binDS[1].sum()]
    # calculate each bin's sum of 0,1,total
    # store values in R*3 ndarray
    for i in range(int(R)):
        subDS = binDS[binDS['bin']==(i+1)]
        N_mat[i][0] = subDS[0].sum()
        N_mat[i][1] = subDS[1].sum()
        N_mat[i][2] = subDS['total'].sum()
    
    # Gini
    if method == 1:
        G_list = [0]*R
        for i in range(int(R)):

            for j in range(2):
                G_list[i] = G_list[i] + N_mat[i][j]*N_mat[i][j]
            G_list[i] = 1 - G_list[i]/(N_mat[i][2]*N_mat[i][2])
        G = 0
        for j in range(2):
            G = G + N_s[j]*N_s[j]
        
        G = 1 - G/(N*N)
        Gr = 0
        for i in range(int(R)):
            Gr = Gr + N_mat[i][2]*(G_list[i]/N)
        M_value = 1 - Gr/G
    # Entropy
    elif method == 2:
        for i in range(int(R)):
            for j in range(2):
                if np.isnan(N_mat[i][j]) or N_mat[i][j] == 0:
                    M_value = 0
        
        E_list = [0]*R
        for i in range(int(R)):
            for j in range(2):
                E_list[i] = E_list[i] - ((N_mat[i][j]/float(N_mat[i][2]))\
                       *np.log(N_mat[i][j]/N_mat[i][2]))
                
            E_list[i] = E_list[i]/np.log(2)#plus
        E = 0
        for j in range(2):
            a = (N_s[j]/N)
            E = E - a*(np.log(a))
            
        E = E/np.log(2)
        Er = 0
        for i in range(2):
            Er = Er + N_mat[i][2]*E_list[i]/N
        M_value = 1 - (Er/E)
        return M_value
    # Pearson X2
    elif method == 3:
        N = N_s[0] + N_s[1]
        X2 = 0
        M = np.empty((R,2))
        for i in range(int(R)):
            for j in range(2):
                M[i][j] = N_mat[i][2]*N_s[j]/N
                X2 = X2 + (N_mat[i][j]-M[i][j]) * (N_mat[i][j]-M[i][j]) / (M[i][j])
        
        M_value = X2
    # Info value
    else:
        if any([isNullZero(N_mat[i][0]),
               isNullZero(N_mat[i][1]),
                isNullZero(N_s[0]),
                isNullZero(N_s[1])]):
            M_value = np.NaN
        else:
            IV =0 
            for i in range(int(R)):
                IV = IV + (N_mat[i][0]/N_s[0] - N_mat[i][1]/N_s[1])\
                    *np.log((N_mat[i][0]*N_s[1])/(N_mat[i][1]*N_s[0]))
            M_value = IV
            
    return M_value


def calCMerit(temp, ix, method,total_num):
    """
    Calculation of the merit function for the current table temp
    ---------------------------------------------
    Params
    temp: pandas dataframe, temp table in _bestSplit 
    ix: single int obj,index of temp, from length of temp 
    method: int obj, metric to split x(1:Gini, 2:Entropy, 3:person chisq, 4:Info value)
    ---------------------------------------------
    Return
    M_value: float or np.nan
    """
    # split data by ix 
    temp_L = temp[temp['i'] <= ix]
    temp_U = temp[temp['i'] > ix]
    # calculate sum of 0, 1, total for each splited data
    n_11 = float(sum(temp_L[0]))
    n_12 = float(sum(temp_L[1]))
    n_21 = float(sum(temp_U[0]))
    n_22 = float(sum(temp_U[1]))
    n_1s = float(sum(temp_L['total']))
    n_2s = float(sum(temp_U['total']))
    # calculate sum of 0, 1 for whole data
    n_s1 = float(sum(temp[0]))
    n_s2 = float(sum(temp[1]))

    N_mat = np.array([[n_11, n_12, n_1s],
                      [n_21, n_22, n_2s]])
    N_s = [n_s1, n_s2]
    # Gini
    if method == 1:
        N = n_1s + n_2s
        G1 = 1- ((n_11*n_11 + n_12*n_12)/float(n_1s*n_1s))
        G2 = 1- ((n_21*n_21 + n_22*n_22)/float(n_2s*n_2s))
        G = 1- ((n_s1*n_s1 + n_s2*n_s2)/float(N*N))
        M_value = 1 - ((n_1s*G1 + n_2s*G2)/float(N*G))
    # Entropy
    elif method == 2:
        N = n_1s + n_2s
        E1= -((n_11/n_1s)*(np.log((n_11/n_1s))) + \
        (n_12/n_1s)*(np.log((n_12/n_1s)))) / (np.log(2)) 
        E2= -((n_21/n_2s)*(np.log((n_21/n_2s))) + \
         (n_22/n_2s)*(np.log((n_22/n_2s))))/(np.log(2)) 
        E = -(((n_s1/N)*(np.log((n_s1/N))) + ((n_s2/N)*\
            np.log((n_s2/N))))/ (np.log(2)))
        M_value = 1-(n_1s*E1 + n_2s*E2)/(N*E)
    # Pearson chisq
    elif method == 3:
        N = n_1s + n_2s
        X2 = 0
        M = np.empty((2,2))
        for i in range(2):
            for j in range(2):
                M[i][j] = N_mat[i][2]*N_s[j]/N
                X2 = X2 + ((N_mat[i][j]-M[i][j])*(N_mat[i][j]-M[i][j]))/M[i][j]
        
        M_value = X2
    # Info Value    
    else:
        if (n_11+n_12) <= total_num or (n_21+n_22) <= total_num :
            M_value = 0
        else:
            IV = ((n_11/n_s1) - (n_12/n_s2)) * np.log((n_11*n_s2)/(n_12*n_s1)) \
             + ((n_21/n_s1) - (n_22/n_s2)) * np.log((n_21*n_s2)/(n_22*n_s1))
            M_value = IV
    return M_value
    

def bestSplit(binDS, method, BinNo,total_num):
    """
    find the best split for one bin dataset
    middle procession functions for _candSplit
    --------------------------------------
    Params
    binDS: pandas dataframe, middle bining table
    method: int obj, metric to split x
        (1:Gini, 2:Entropy, 3:person chisq, 4:Info value)
    BinNo: int obj, bin number of binDS
    --------------------------------------
    Return
    newbinDS: pandas dataframe
    """
    binDS = binDS.sort_values(by=['bin','pdv1'])  
    mb = len(binDS[binDS['bin']==BinNo])
       
    bestValue = 0
    bestI = 1
    for i in range(1, mb):
        # split data by i
        # metric: Gini,Entropy,pearson chisq,Info value 
        value = calCMerit(binDS, i,method,total_num=total_num)
        # if value>bestValue，then make value=bestValue，and bestI = i
        if bestValue < value:
            bestValue = value
            bestI = i
    # create new var split
    binDS['split'] = np.where(binDS['i'] <= bestI, 1, 0)
    binDS = binDS.drop('i', axis=1)
    newbinDS = binDS.sort_values(by=['split','pdv1'])
    # rebuild var i
    newbinDS_0 = newbinDS[newbinDS['split']==0]
    newbinDS_1 = newbinDS[newbinDS['split']==1]
    newbinDS_0['i'] = range(1, len(newbinDS_0)+1)
    newbinDS_1['i'] = range(1, len(newbinDS_1)+1)
    newbinDS = pd.concat([newbinDS_0, newbinDS_1], axis=0)
    return newbinDS#.sort_values(by=['split','pdv1'])


def candSplit(binDS, method,total_num):
    """
    Generate all candidate splits from current Bins 
    and select the best new bins
    middle procession functions for binContVar & reduceCats
    ---------------------------------------------
    Params
    binDS: pandas dataframe, middle bining table
    method: int obj, metric to split x
        (1:Gini, 2:Entropy, 3:person chisq, 4:Info value)
    --------------------------------------------
    Return
    newBins: pandas dataframe, split results
    """
    # sorted data by bin&pdv1
    binDS = binDS.sort_values(by=['bin','pdv1'])    
    # get the maximum of bin
    Bmax = max(binDS['bin'])
    # screen data and cal nrows by diffrence bin
    # and save the results in dict
    temp_binC = dict()
    m = dict()
    for i in range(1, Bmax+1):
        temp_binC[i] = binDS[binDS['bin']==i]
        m[i] = len(temp_binC[i])
    """
    CC
    """
    # create null dataframe to save info
    temp_trysplit = dict()
    temp_main = dict()
    bin_i_value = []
    for i in range(1, Bmax+1):
        if m[i] > 1: # if nrows of bin > 1
            # split data by best i        
            temp_trysplit[i] = bestSplit(temp_binC[i], method, i,total_num=total_num)            
            temp_trysplit[i]['bin'] = np.where(temp_trysplit[i]['split']==1, 
                                               Bmax+1, 
                                               temp_trysplit[i]['bin'])
            # delete bin == i
            temp_main[i] = binDS[binDS['bin']!=i]
            # vertical combine temp_main[i] & temp_trysplit[i]
            temp_main[i] = pd.concat([temp_main[i],temp_trysplit[i]], axis=0)
            # calculate metric of temp_main[i]
            value = Gvalue(temp_main[i], method)
            newdata = [i, value]
            bin_i_value.append(newdata)
    #find maxinum of value bintoSplit
    bin_i_value.sort(key=lambda x:x[1], reverse=True)
    #binNum = temp_all_Vals['BinToSplit']
    binNum = bin_i_value[0][0]
    newBins = temp_main[binNum].drop('split', axis=1)
    return newBins.sort_values(by=['bin', 'pdv1']) 


def EqualWidthBinMap(x, Acc, adjust):
    """
    Data bining function, 
    middle procession functions for binContVar
    method: equal width
    Mind: Generate bining width and interval by Acc
    --------------------------------------------
    Params
    x: pandas Series, data need to bining
    Acc: float less than 1, partition ratio for equal width bining
    adjust: float or np.inf, bining adjust for limitation
    --------------------------------------------
    Return
    bin_map: pandas dataframe, Equal width bin map
    """
    varMax = x.max()
    varMin = x.min()
    # generate range by Acc
    Mbins = int(1./Acc)
    minMaxSize = (varMax - varMin)/Mbins
    # get upper_limit and loewe_limit
    ind = range(1, Mbins+1)
    Upper = pd.Series(index=ind, name='upper')
    Lower = pd.Series(index=ind, name='lower')
    for i in ind:
        Upper[i] = varMin + i*minMaxSize
        Lower[i] = varMin + (i-1)*minMaxSize
    
    # adjust the min_bin's lower and max_bin's upper     
    Upper[Mbins] = Upper[Mbins]+adjust
    Lower[1] = Lower[1]-adjust
    bin_map = pd.concat([Lower, Upper], axis=1)
    bin_map.index.name = 'bin'
    return bin_map    


def applyBinwoe(x, bin_map,det="woe"):
    """
    Generate result of bining by bin_map
    ------------------------------------------------
    Params
    x: pandas Series
    bin_map: pandas dataframe, map table
    ------------------------------------------------
    Return
    bin_res: pandas Series, result of bining
    """
    bin_res = np.array([0] * x.shape[-1], dtype=float)
    bin_map=bin_map.set_index(det)
    for i in bin_map.index:
        upper = bin_map['max'][i]
        lower = bin_map['min'][i]
        x1 = x[np.where((x >= lower) & (x <= upper))[0]]
        mask = np.in1d(x, x1)
        bin_res[mask] = i
    
    bin_res = pd.Series(bin_res, index=x.index)
    bin_res.name = x.name
    return bin_res

def applyBinMap(x, bin_map):
    """
    Generate result of bining by bin_map
    ------------------------------------------------
    Params
    x: pandas Series
    bin_map: pandas dataframe, map table
    ------------------------------------------------
    Return
    bin_res: pandas Series, result of bining
    """
    bin_res = np.array([0] * x.shape[-1], dtype=int)
    
    for i in bin_map.index:
        upper = bin_map['upper'][i]
        lower = bin_map['lower'][i]
        x1 = x[np.where((x >= lower) & (x <= upper))[0]]
        mask = np.in1d(x, x1)
        bin_res[mask] = i
    
    bin_res = pd.Series(bin_res, index=x.index)
    bin_res.name = x.name + "_BIN"
    
    return bin_res

def _combineBins(temp_cont, target):
    """
    merge all bins that either 0 or 1 or total =0
    middle procession functions for binContVar 
    ---------------------------------
    Params
    temp_cont: pandas dataframe, middle results of binContVar
    target: target label
    --------------------------------
    Return
    temp_cont: pandas dataframe
    修改： 如果有0值 就和后面一条合并
    """
    drop_in=[]
    for i in range(len(temp_cont.index)):
        rowdata = temp_cont.iloc[i,:]
        
        if i == len(temp_cont.index)-1:
            ix=i-1
            while any(temp_cont.iloc[ix,:3] == 0):
                ix=ix-1
        else:
            ix=i+1
        if any(rowdata[:3] == 0):#如果0,1,total有一项为0，则运行
            #

            temp_cont.iloc[ix, target] = temp_cont.iloc[ix, target] + rowdata[target]
            temp_cont.iloc[ix, 0] = temp_cont.iloc[ix, 0] + rowdata[0]
            temp_cont.iloc[ix, 2] = temp_cont.iloc[ix, 2] + rowdata[2]
            drop_in.append(temp_cont.index[i])
            #
            if i < temp_cont.index.max():
                temp_cont.iloc[ix,3] = rowdata['lower']
            else:
                temp_cont.iloc[ix,4] = rowdata['upper']
    temp_cont = temp_cont.drop(drop_in, axis=0)
            
        
    return temp_cont.sort_values(by='pdv1')


def getNewBins(sub, i):
    """
    get new lower, upper, bin, total for sub
    middle procession functions for binContVar
    -----------------------------------------
    Params
    sub: pandas dataframe, subdataframe of temp_map
    i: int, bin number of sub
    ----------------------------------------
    Return
    df: pandas dataframe, one row
    """
    l = len(sub)
    total = sub['total'].sum()
    bad = sub.iloc[:,1].sum()
    good = sub.iloc[:,0].sum()
    first = sub.iloc[0,:]
    last = sub.iloc[l-1,:]
    
    lower = first['lower']
    upper = last['upper']
    df = pd.DataFrame()
    df = df.append([i, lower, upper,good,bad,total], ignore_index=True).T
    df.columns = ['bin', 'lower', 'upper','good','bad','total']
    return df


def binContVar(x, y, method, mmax=5, Acc=0.01, target=1, adjust=0.0001):
    """
    Optimal binings for contiouns var x by (y & method)
    method is represent by number, 
        1:Gini, 2:Entropy, 3:person chisq, 4:Info value
    ---------------------------------------------
    Params
    x: pandas Series, which need to reduce category
    y: pandas Series, 0-1 distribute dependent variable
    method: int obj, metric to split x
    mmax: int, bining number 
    Acc: float less than 1, partition ratio for equal width bining
    badlabel: target label
    adjust: float or np.inf, bining adjust for limitation
    ---------------------------------------------
    Return
    temp_Map: pandas dataframe, Optimal bining map
    """
    # if y is not 0-1 binary variable, then raise a error
    check_target_binary(y)
    # data bining by Acc, method: width equal
    bin_map = EqualWidthBinMap(x, Acc, adjust=adjust)
    # mapping x to bin number and combine with x&y
    bin_res = applyBinMap(x, bin_map)
    temp_df = pd.concat([x, y, bin_res], axis=1)
    # calculate freq of 0, 1 in y group by bin_res
    t1 = pd.crosstab(index=temp_df[bin_res.name], columns=y)
    # calculate freq of bin, and combine with t1
    t2 = temp_df.groupby(bin_res.name).count().ix[:,0]
    t2 = pd.DataFrame(t2)
    t2.columns = ['total'] 
    t = pd.concat([t1, t2], axis=1)
    # merge t & bin_map by t,
    # if all(0,1,total) == 1, so corresponding row will not appear in temp_cont
    temp_cont = pd.merge(t, bin_map, 
                         left_index=True, right_index=True, 
                         how='left')
    temp_cont['pdv1'] = temp_cont.index
    # if any(0,1,total)==0, then combine it with per bin or next bin
    temp_cont = _combineBins(temp_cont, target)
    # calculate other temp vars      
    temp_cont['bin'] = 1
    temp_cont['i'] = range(1, len(temp_cont)+1)
    temp_cont['var'] = temp_cont.index
    total_num = round(len(y)*0.05)
    nbins = 1
    # exe candSplit mmax times
    while(nbins < mmax):
        temp_cont = candSplit(temp_cont, method=method,total_num=total_num)
        nbins += 1
       
    temp_cont = temp_cont.rename(columns={'var':'oldbin'})
    temp_Map1 = temp_cont.drop(['pdv1' , 'i'], axis=1)
    temp_Map1 = temp_Map1.sort_values(by=['bin', 'oldbin'])
    # get new lower, upper, bin, total for sub
    data = pd.DataFrame()
    s = set()
    for i in temp_Map1['bin']:
        if i in s:
            pass
        else:
            sub_Map = temp_Map1[temp_Map1['bin']==i]
            rowdata = getNewBins(sub_Map, i)
            data = data.append(rowdata, ignore_index=True)
            s.add(i)
    
    # resort data
    data = data.sort_values(by='lower')
    data['newbin'] = range(1, mmax+1)
    data = data.drop('bin', axis=1)
    data.index = data['newbin']
    return data


def groupCal(x, y, badlabel=1):
    """
    group calulate for x by y
    middle proporcessing function for reduceCats
    -------------------------------------
    Params
    x: pandas Series, which need to reduce category
    y: pandas Series, 0-1 distribute dependent variable
    badlabel: target label
    ------------------------------------
    Return
    temp_cont: group calulate table
    m: nrows of temp_cont
    """
    
    temp_cont = pd.crosstab(index=x, columns=y, margins=False)
    temp_cont['total'] = temp_cont.sum(axis=1)
    temp_cont['pdv1'] = temp_cont[badlabel]/temp_cont['total']
    
    temp_cont['i']= range(1, temp_cont.shape[0]+1)
    temp_cont['bin'] = 1
    m = temp_cont.shape[0]
    return temp_cont, m


def reduceCats(x, y,  method=1, mmax=5, badlabel=1):
    """
    Reduce category for x by y & method
    method is represent by number, 
        1:Gini, 2:Entropy, 3:person chisq, 4:Info value
    ----------------------------------------------
    Params:
    x: pandas Series, which need to reduce category
    y: pandas Series, 0-1 distribute dependent variable
    method: int obj, metric to split x
    mmax: number to reduce
    badlabel: target label
    ---------------------------------------------
    Return
    temp_cont: pandas dataframe, reduct category map
    """
    check_target_binary(y)
    temp_cont, m = groupCal(x, y, badlabel=badlabel) 
    nbins = 1
    while(nbins< mmax):
        temp_cont =candSplit(temp_cont, method=method)
        nbins += 1
    
    temp_cont = temp_cont.rename(columns={'var':x.name})
    temp_cont = temp_cont.drop([0, 1, 'i', 'pdv1'], axis=1)
    return temp_cont.sort_values(by='bin')


def applyMapCats(x, bin_map):
    """
    convert x to newbin by bin_map
    ------------------------------
    Params
    x: pandas Series
    bin_map: pandas dataframe, mapTable contain new bins
    ------------------------------
    Return
    new_x: pandas Series, convert results
    """
    d = dict()
    for i in bin_map.index:
        subData = bin_map[bin_map.index==i]
        value = subData.ix[i,'bin']
        d[i] = value
    
    new_x = x.map(d)
    new_x.name = x.name+'_BIN'
    return new_x


def tableTranslate(red_map):
    """
    table tranlate for red_map
    ---------------------------
    Params
    red_map: pandas dataframe,reduceCats results
    ---------------------------
    Return
    res: pandas series
    """
    l = red_map['bin'].unique()
    res = pd.Series(index=l)
    for i in l:
        value = red_map[red_map['bin']==i].index
        value = list(value.map(lambda x:str(x)+';'))
        value = "".join(value)
        res[i] = value
    return res



"""
    --------------------------------------------------------------
    等频分箱--frequency_cut
    
"""
def frequency_cut(series,bin, return_point=True):
    """
    返回每个值所属的箱，如果return_point=True，返回每个切分的点
    由于样本的分布不同，可能会出现bin减少的情况(总箱数少于指定的bin),区间是前开后闭,箱数从1开始
    split_point = [2, 3, 5], 则区间bin为4个,(-inf, 2],(2,3],(3,5],(5, +inf]
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    edges = np.array([float(i) / bin for i in range(bin + 1)])

    # 返回第一个edges的rank大于实际rank的索引
    bin_result = np.array(series.rank(pct=1).apply(lambda x: (edges >= x).argmax()))
    split_point = [max(series[bin_result == point]) for point in np.unique(bin_result)[:-1]]
    if len(np.unique(bin_result)) < bin:
        bin_set = np.unique(bin_result)
        replace_dict = {bin_set[i]: i+1 for i in range(len(bin_set)) if bin_set[i] != i+1}
        for k, v in replace_dict.items():
            bin_result[bin_result == k] = v
    
    if return_point:
        return bin_result, split_point
    else:
        return bin_result

    


def caliv_dp(data,group,ycol=-1):

#'''简单的等频数据拆分'''
    all_iv_detail = pd.DataFrame([])
    if type(ycol) == int:
        ycol = data.columns[ycol]
    y=data[ycol]
    if type(group) == int:
        column_names = data.columns[data.columns != ycol]
    elif isinstance(group,pd.DataFrame):
        column_names = group['var_name'].unique()
    else:
        print("argument 'group' type is wrong")
        return 0,0  
    
    for i in column_names: 
        print(i)     
        aa,bb = frequency_cut(data[i],group)  
        bad_sample = y.sum()
        good_sample = y.count()-y.sum()
        a1=pd.DataFrame(aa,columns=[i]) 
        tmp = pd.concat([a1,y.reset_index(drop=True)],axis=1) #tmp是临时的iv计算数据框
        grouped = tmp.groupby(i)["y"]
        cols=pd.DataFrame()
        cols['rank'] = range(len(grouped))
        cols['bad'] = grouped.sum().reset_index(drop=True)
        cols['good'] = grouped.count().reset_index(drop=True)-grouped.sum().reset_index(drop=True)
        cols['max'] =  pd.Series(bb)
        cols['min'] =  pd.Series(bb).shift(1)
        cols['BAD_rate'] = cols['bad'] / bad_sample
        cols['GOOD_rate'] = cols['good'] / good_sample
        cols['total'] = ( cols['good']+ cols['bad'])
        cols["total_rate"]=cols['total']/y.count()
        cols["rate"]=grouped.mean().reset_index(drop=True)
        cols['woe'] = np.log(cols['BAD_rate']/cols['GOOD_rate'])
        cols.ix[cols['woe'] == np.inf,'woe'] = 0 # 分母为0的先设置为0
        cols.ix[cols['woe'] == -np.inf,'woe'] = 0 # 分母为0的先设置为0
        cols['MIV'] = (cols['BAD_rate']-cols['GOOD_rate'])*cols['woe']
        cols['ori_IV'] = cols['MIV'].sum()
        cols['var_name'] = i     
        all_iv_detail=all_iv_detail.append(cols)
    all_iv_detail = all_iv_detail.sort_values(by=['ori_IV','var_name','max'],ascending=[False,True,True])
    return all_iv_detail  


