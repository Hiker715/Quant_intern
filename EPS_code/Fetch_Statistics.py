# -*- coding: utf-8 -*-
"""
Created on Sat Aug  4 14:24:48 2018

@author: sy
"""
# 目标：在统计上检验显不显著
# epsAvg计算的是滞后3、5、7、14、21天的变化情况
# return计算的是间隔1、3、5、10、20天的收益情况
# 取每天eps<0的样本中各个累计收益的中位数，只能研究是否显著小于0
# 要研究eps和收益的相关性，则需要计算每天的相关系数。计算rankIC

import pandas as pd
import numpy as np
import pymssql
import datetime as dt

path_eR = "F:\\Intern\\3outputFile\\EPS_excessR\\excess_return"

def get_DB_t_days(start, end, conn):
    
    """
    This function will fetch all trading days from db , as string
    """
    
    query = """
        SELECT TradingDate
        FROM JYDB..QT_TradingDayNew
        WHERE TradingDate >= '%s' AND TradingDate <= '%s' AND IfTradingDay = 1 AND SecuMarket=83
    """ % (start, end)

    t_days = pd.read_sql(query, conn)
    t_days = t_days["TradingDate"].apply(lambda x:dt.datetime.strftime(x,'%Y%m%d'))
    return np.array(t_days)

# 研究指定区间内的交易日
conn =pymssql.connect(server='172.16.37.140', user='jydb', password='jydb', port=1433, database='JYDB')
tradingday = get_DB_t_days('20100201', '20180731', conn)
fulldates = tradingday
eps_lag = [3,5,7,10,15,20]
er_col = ['excess_r_1', 'excess_r_3', 'excess_r_5', 'excess_r_10', 'excess_r_15','excess_r_20', 'excess_r_25','excess_r_30']



# 每天只取一个统计量
def fetch_statistics(fulldates, sta_func, eps_threshold, er_path, er_col=er_col):
    """
    columns: date, 'excess_r_1', 'excess_r_3', 'excess_r_5', 'excess_r_10', 'excess_r_20'
    """

    stsdf = pd.DataFrame(index = fulldates)

    for loopdate in fulldates:
        looper = pd.read_csv(er_path+'\\data_%s.csv'%loopdate)
        looper = looper[looper.EPS_change<eps_threshold]
        for col in er_col:
            stsdf.loc[loopdate, col] = sta_func(looper[col])
    return stsdf.reset_index()


median_data = pd.DataFrame()
# clolumns: date, lag, 'excess_r_1', 'excess_r_3', 'excess_r_5', 'excess_r_10', 'excess_r_20'
for lag in eps_lag:
    pather = path_eR + '\\Factor_Exposure_eps_lag%s'%lag
    temp_data = fetch_statistics(fulldates, np.nanmedian, 0, pather)
    temp_data['lag'] = lag
    median_data = pd.concat([median_data, temp_data])

median_data.to_csv("F:\\Intern\\3outputFile\\AggData\\Median_lag3571421", index=False)



# 每天各个行业都取一个统计量
def fetch_industry_statistics(fulldates, sta_func, eps_threshold, er_path, er_col=er_col):
    """
    columns: date, industry, 'excess_r_1', 'excess_r_3', 'excess_r_5', 'excess_r_10', 'excess_r_20'
    """

    stsdf = pd.DataFrame()
    for loopdate in fulldates:
        looper = pd.read_csv(er_path+'\\data_%s.csv'%loopdate)
        looper = looper[looper.EPS_change<eps_threshold]
        temp_data = looper.groupby('XSIndustry')[er_col].agg(sta_func).reset_index()
        temp_data['date'] = loopdate
        stsdf = pd.concat([stsdf, temp_data])

    return stsdf

median_indu_data = pd.DataFrame()
# clolumns: date, lag, XSIndustry, 'excess_r_1', 'excess_r_3', 'excess_r_5', 'excess_r_10', 'excess_r_20'
for lag in eps_lag:
    pather = path_eR + '\\Factor_Exposure_eps_lag%s'%lag
    temp_data = fetch_industry_statistics(fulldates, np.nanmedian, 0, pather)
    temp_data['lag'] = lag
    median_indu_data = pd.concat([median_indu_data, temp_data])
median_indu_data.to_csv("F:\\Intern\\3outputFile\\AggData\\Median_indu_lag3571421", index=False)



# 每天取十个统计量
def fetch_ten_statistics(fulldates, sta_func, eps_threshold, er_path, er_col=er_col):
    """
    columns: date, class, 'excess_r_1', 'excess_r_3', 'excess_r_5', 'excess_r_10', 'excess_r_20'
    """
    
    stsdf = pd.DataFrame()
    for loopdate in fulldates:
        temp_data = pd.DataFrame(index = np.arange(1,11), columns=er_col)
        looper = pd.read_csv(er_path+'\\data_%s.csv'%loopdate)
        looper = looper[looper.EPS_change<eps_threshold]
        looper = looper.sort_values(by='EPS_change', ascending=False).reset_index(drop=True)
        n = len(looper)
        m = int(n/10)
        for i in range(9):
            temp_data.loc[i+1][er_col] = looper.iloc[i*m:(i+1)*m][er_col].agg(sta_func)
        temp_data.loc[10][er_col] = looper.iloc[9*m: ][er_col].agg(sta_func)
        temp_data.reset_index(inplace=True)
        temp_data['date'] = loopdate
        stsdf = pd.concat([stsdf, temp_data])

    return stsdf

median_ten_data = pd.DataFrame()
for lag in eps_lag:
    pather = path_eR + '\\Factor_Exposure_eps_lag%s'%lag
    temp_data = fetch_ten_statistics(fulldates, np.nanmedian, 0, pather)
    temp_data['lag'] = lag
    median_ten_data = pd.concat([median_ten_data, temp_data])
median_ten_data.rename(columns = {'index':'class_'}, inplace=True)
median_ten_data.to_csv("F:\\Intern\\3outputFile\\AggData\\Median_ten_lag3571421", index=False)
 
#  RuntimeWarning: All-NaN slice encountered
a = np.array([np.nan,np.nan, np.nan])
np.nanmedian(a)

# 检查temp_data.loc[i] = looper.iloc[i*m:(i+1)*m][er_col].agg(sta_func)
looper = pd.read_csv("F:\\Intern\\3outputFile\\EPS_excessR\\excess_return\\Factor_Exposure_eps_lag5\\data_20100609.csv")
er_col = ['excess_r_1', 'excess_r_3', 'excess_r_5', 'excess_r_10', 'excess_r_20']
temp_data = pd.DataFrame(index = np.arange(1,11), columns=er_col)
temp_data.loc[1][er_col] =  looper.iloc[1*200:(1+1)*200][er_col].agg(np.nanmedian)
a = looper.iloc[1*200:(1+1)*200][er_col].agg(np.nanmedian)



# 计算rankic
from scipy.stats import rankdata
from scipy.stats import spearmanr 
# Calculates a Spearman rank-order correlation coefficient and the p-value to test for non-correlation.


def fetch_positive_rankIC(fulldates, er_path, er_col=er_col):
    """
    以天为单位计算rankic
    """
    stsdf = pd.DataFrame(index=fulldates)
    for loopdate in fulldates:
        looper = pd.read_csv(er_path+'\\data_%s.csv'%loopdate)
        looper = looper[looper.EPS_change>0]
        looper['eps_rank'] = rankdata(looper['EPS_change'], method='dense')
        for col in er_col:
            looper['%s_rank'%col] = rankdata(looper[col], method='dense')
            stsdf.loc[loopdate, 'rankic_%s'%col], stsdf.loc[loopdate, 'pValue_%s'%col] = spearmanr(looper['eps_rank'], looper['%s_rank'%col]) 
            
    return stsdf


def fetch_negative_rankIC(fulldates, er_path, er_col=er_col):
    
    stsdf = pd.DataFrame(index=fulldates)
    for loopdate in fulldates:
        looper = pd.read_csv(er_path+'\\data_%s.csv'%loopdate)
        looper = looper[looper.EPS_change<0]
        looper['eps_rank'] = rankdata(looper['EPS_change'], method='dense')
        for col in er_col:
            looper['%s_rank'%col] = rankdata(looper[col], method='dense')
            stsdf.loc[loopdate, 'rankic_%s'%col], stsdf.loc[loopdate, 'pValue_%s'%col] = spearmanr(looper['eps_rank'], looper['%s_rank'%col]) 
            
    return stsdf

rankic_p_data = pd.DataFrame()
for lag in eps_lag:
    pather = path_eR + '\\Factor_Exposure_eps_lag%s'%lag
    temp_data = fetch_positive_rankIC(fulldates, pather)
    temp_data['lag'] = lag
    rankic_p_data = pd.concat([rankic_p_data, temp_data])
rankic_p_data = rankic_p_data.reset_index().rename(columns={'index':'date'})   
rankic_n_data = pd.DataFrame()
for lag in eps_lag:
    pather = path_eR + '\\Factor_Exposure_eps_lag%s'%lag
    temp_data = fetch_negative_rankIC(fulldates, pather)
    temp_data['lag'] = lag
    rankic_n_data = pd.concat([rankic_n_data, temp_data])
rankic_n_data = rankic_n_data.reset_index().rename(columns={'index':'date'}) 

rankic_n_data.to_csv("F:\\Intern\\2CompanyFile\\EPStask\\rankic_n_data.csv", index=False)
rankic_p_data.to_csv("F:\\Intern\\2CompanyFile\\EPStask\\rankic_p_data.csv", index=False)

# 检验rankic, eps_change有相当多0
looper = pd.read_csv("F:\\Intern\\3outputFile\\EPS_excessR\\excess_return\\Factor_Exposure_eps_lag5\\data_20100913.csv")
looper['eps_rank'] = rankdata(looper['EPS_change'])
stsdf = pd.DataFrame(index=fulldates)
for col in er_col:
    looper['%s_rank'%col] = rankdata(looper[col], method='dense')
    stsdf.loc['20100913', 'rankic_%s'%col] ,stsdf.loc['20100913', 'pValue_%s'%col]= spearmanr(looper['eps_rank'], looper['%s_rank'%col]) 

        


# 生成研究
# 1、eps覆盖率
eps_rn = pd.read_table("F:\\Intern\\3outputFile\\outTXT\\epsRowNumber.txt", header=None, names=['fn', 'rn'], delimiter='    ')
sto_rn = pd.read_table("F:\\Intern\\3outputFile\\outTXT\\stockRowNumber.txt", header=None, names=['fn', 'rn'], delimiter='    ')
eps_rn.fn = eps_rn['fn'].apply(lambda s: s[5:13])
sto_rn.fn =sto_rn['fn'].apply(lambda s: s.split('.')[-1])
eps_rn = eps_rn.merge(pd.DataFrame(fulldates, columns=['fn']), how='right')
sto_rn = sto_rn.merge(pd.DataFrame(fulldates, columns=['fn']), how='right')
eps_rn = eps_rn.merge(sto_rn, suffixes=['eps','sto'], on='fn')
eps_rn['eps_ratio'] = eps_rn.rneps/eps_rn.rnsto

import matplotlib.pyplot as plt
import matplotlib as mpl
from dateutil.parser import parse

# =============================================================================
# start = parse(fulldates[0])
# stop = parse(fulldates[-1])
# delta=dt.timedelta(1)
# =============================================================================
# 返回浮点型的日期序列，这个是生成时间序列
# dates=mpl.dates.drange(start,stop,delta)
dates=[parse(d) for d in fulldates]
values=eps_rn['eps_ratio']
#存在两个问题，一个是坐标轴没有按照日期的形式去标注，另一个是刻度的数量和位置也不合适
fig=plt.figure(figsize=(24,12))#调整画图空间的大小
plt.plot(dates,values,linestyle='-',marker='*',c='r',alpha=0.5)#作图
ax=plt.gca()
date_format=mpl.dates.DateFormatter('%Y-%m')#设定显示的格式形式
ax.xaxis.set_major_formatter(date_format)#设定x轴主要格式
ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(90))#设定坐标轴的显示的刻度间隔
fig.autofmt_xdate()#防止x轴上的数据重叠，自动调整。
plt.savefig("F:\\Intern\\3outputFile\\Figure\\eps_percent.png")


# 2、分成十个挡位研究，(之研究eps为负的)而且时间也分区间研究，比如以半年为一个区间
# 一共102个自然月，以半年为单位，将交易日划分成十七个区间


def fetch_half_year_ten_statistics(fulldates, sta_func, eps_threshold, er_path, er_col=er_col):
    """
    columns: date, class, 'excess_r_1', 'excess_r_3', 'excess_r_5', 'excess_r_10', 'excess_r_20'
    """
    
    stsdf = pd.DataFrame()
    # 产生17个区间的端点
    seq = np.linspace(0,len(fulldates),18)
    for i in range(17):
        temp_data = pd.DataFrame()
        for loopdate in fulldates[int(seq[i]):int(seq[i+1])]:
            looper = pd.read_csv(er_path+'\\data_%s.csv'%loopdate)
            looper = looper[looper.EPS_change<eps_threshold]
            temp_data = pd.concat([temp_data, looper])
            
        temp_percentile = np.percentile(temp_data['EPS_change'],[0,10,20,30,40,50,60,70,80,90,100])
        temp_percentile[0] = temp_percentile[0] - 1
        temp_data['class_'] = pd.cut(temp_data['EPS_change'], temp_percentile, labels=[10,9,8,7,6,5,4,3,2,1])
        temp_data = temp_data.groupby(['EndDate','class_'])[er_col].agg(sta_func).reset_index().rename(columns={'EndDate':'date'})
        stsdf = pd.concat([stsdf, temp_data])
    return stsdf

median_half_year_ten_data = pd.DataFrame()
for lag in eps_lag:
    pather = path_eR + '\\Factor_Exposure_eps_lag%s'%lag
    temp_data = fetch_half_year_ten_statistics(fulldates, np.nanmedian, 0, pather)
    temp_data['lag'] = lag
    median_half_year_ten_data = pd.concat([median_half_year_ten_data, temp_data])
median_half_year_ten_data['class_'] = median_half_year_ten_data['class_'].astype(int)
median_half_year_ten_data.to_csv("F:\\Intern\\2CompanyFile\\EPStask\\median_half_year_ten_data.csv", index=False)

a = median_half_year_ten_data[median_half_year_ten_data.class_==10]


# 3、RankIC
# 散点图看不出规律
for l in [3,5,7,14,21]:
    rank_list = []
    fig=plt.figure(figsize=(24,12))
    for col in er_col:
        rank_list.append('rankic_%s'%col)
        values=rankic_n_data.loc[rankic_n_data.lag==l, 'rankic_%s'%col]
        plt.plot(dates,values,linestyle='-',marker='*',c='r',alpha=0.5)
    ax=plt.gca()
    date_format=mpl.dates.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(date_format)
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(60))#设定坐标轴的显示的刻度间隔
    fig.autofmt_xdate()#防止x轴上的数据重叠，自动调整。
    plt.savefig("F:\\Intern\\3outputFile\\Figure\\rankic_lag%s.png"%l)

fig=plt.figure(figsize=(24,12))
values=rankic_n_data.loc[rankic_n_data.lag==5, 'rankic_excess_r_1']
plt.plot(dates,values,linestyle='-',marker='*',c='r',alpha=0.5)
ax=plt.gca()
date_format=mpl.dates.DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_formatter(date_format)
ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(90))#设定坐标轴的显示的刻度间隔
fig.autofmt_xdate()#防止x轴上的数据重叠，自动调整。
plt.savefig("F:\\Intern\\3outputFile\\Figure\\rankic_lag5_r_1.png")

# rankic为正的概率, rankic均值，IR，查看p值不显著的比例
# change为负
rank_list = []
for col in er_col:
    rank_list.append('rankic_%s'%col)
    
pValue_list = []
for col in er_col:
    pValue_list.append('pValue_%s'%col)
    
ic_n_positive_rate = pd.DataFrame(index=eps_lag, columns=rank_list)
for l in eps_lag:
    for r in rank_list:
        ic_n_positive_rate.loc[l,r] = sum(rankic_n_data.loc[rankic_n_data.lag==l, r]>0)/len(rankic_n_data.loc[rankic_n_data.lag==l, r])

ic_n_pValue_rate = pd.DataFrame(index=eps_lag, columns=pValue_list)
for l in eps_lag:
    for r in pValue_list:
        ic_n_pValue_rate.loc[l,r] = sum(rankic_n_data.loc[rankic_n_data.lag==l, r]<0.05)/len(rankic_n_data.loc[rankic_n_data.lag==l, r])

ic_n_mean = pd.DataFrame(index=eps_lag, columns=rank_list)
for l in eps_lag:
    for r in rank_list:
        ic_n_mean.loc[l,r] = np.mean(rankic_n_data.loc[rankic_n_data.lag==l, r])
        
ic_n_ir = pd.DataFrame(index=eps_lag, columns=rank_list)
for l in eps_lag:
    for r in rank_list:
        ic_n_ir.loc[l,r] = np.mean(rankic_n_data.loc[rankic_n_data.lag==l, r])/np.std(rankic_n_data.loc[rankic_n_data.lag==l, r])
ic_n_ir = ic_n_ir * np.sqrt(250)

# change为正
ic_p_pValue_rate = pd.DataFrame(index=eps_lag, columns=pValue_list)
for l in eps_lag:
    for r in pValue_list:
        ic_p_pValue_rate.loc[l,r] = sum(rankic_p_data.loc[rankic_p_data.lag==l, r]<0.05)/len(rankic_p_data.loc[rankic_p_data.lag==l, r])

ic_p_positive_rate = pd.DataFrame(index=eps_lag, columns=rank_list)
for l in eps_lag:
    for r in rank_list:
        ic_p_positive_rate.loc[l,r] = sum(rankic_p_data.loc[rankic_p_data.lag==l, r]>0)/len(rankic_p_data.loc[rankic_p_data.lag==l, r])

ic_p_mean = pd.DataFrame(index=eps_lag, columns=rank_list)
for l in eps_lag:
    for r in rank_list:
        ic_p_mean.loc[l,r] = np.mean(rankic_p_data.loc[rankic_p_data.lag==l, r])
        
ic_p_ir = pd.DataFrame(index=eps_lag, columns=rank_list)
for l in eps_lag:
    for r in rank_list:
        ic_p_ir.loc[l,r] = np.mean(rankic_p_data.loc[rankic_p_data.lag==l, r])/np.std(rankic_p_data.loc[rankic_p_data.lag==l, r])
ic_p_ir = ic_p_ir * np.sqrt(250)








# 计算rankic移动平均，并对其中一个作图观察
ic_ma = rankic_n_data.copy()
for l in eps_lag:
    for r in rank_list:
        ic_ma.loc[rankic_n_data.lag==l, r] = rankic_n_data.loc[rankic_n_data.lag==l, r].rolling(30).mean()

fig=plt.figure(figsize=(24,12))
values=ic_ma.loc[ic_ma.lag==5, 'rankic_excess_r_1']
plt.plot(dates,values,linestyle='-',marker='*',c='r',alpha=0.5)
ax=plt.gca()
date_format=mpl.dates.DateFormatter('%Y-%m')
ax.xaxis.set_major_formatter(date_format)
ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(60))#设定坐标轴的显示的刻度间隔
fig.autofmt_xdate()#防止x轴上的数据重叠，自动调整。
plt.savefig("F:\\Intern\\3outputFile\\Figure\\ic_ma_lag5_r1.png")

# 4、对不同class的eps计算统计量均值、中位数、方差
# columns: date, class_, 'excess_r_1', 'excess_r_3', 'excess_r_5', 'excess_r_10', 'excess_r_20'
eps_class_mean = median_half_year_ten_data.groupby('class_')[er_col].agg(np.nanmean)
eps_class_median = median_half_year_ten_data.groupby('class_')[er_col].agg(np.nanmedian)
eps_class_std = median_half_year_ten_data.groupby('class_')[er_col].agg(np.std)

eps_lag_class_mean = median_half_year_ten_data.groupby(['lag','class_'])[er_col].agg(np.nanmean).reset_index().sort_values(by=['lag','class_'])
eps_lag_class_median = median_half_year_ten_data.groupby(['lag','class_'])[er_col].agg(np.nanmedian).reset_index().sort_values(by=['lag','class_'])
eps_lag_class_std = median_half_year_ten_data.groupby(['lag','class_'])[er_col].agg(np.std).reset_index().sort_values(by=['lag','class_'])

# =============================================================================
# np.nanmean(median_half_year_ten_data.loc[(median_half_year_ten_data['lag']==3) & (median_half_year_ten_data['class_']==1), 'excess_r_1'])
# np.nanmean(median_half_year_ten_data.loc[(median_half_year_ten_data['lag']==5) & (median_half_year_ten_data['class_']==1), 'excess_r_1'])
# np.nanmean(median_half_year_ten_data.loc[(median_half_year_ten_data['lag']==7) & (median_half_year_ten_data['class_']==1), 'excess_r_1'])
# np.nanmean(median_half_year_ten_data.loc[(median_half_year_ten_data['lag']==10) & (median_half_year_ten_data['class_']==1), 'excess_r_1'])
# np.nanmean(median_half_year_ten_data.loc[(median_half_year_ten_data['lag']==15) & (median_half_year_ten_data['class_']==1), 'excess_r_1'])
# 
# a = median_half_year_ten_data[(median_half_year_ten_data['lag']==3) & (median_half_year_ten_data['class_']==1)]
# b = median_half_year_ten_data[(median_half_year_ten_data['lag']==5) & (median_half_year_ten_data['class_']==1)]
# c = median_half_year_ten_data[(median_half_year_ten_data['lag']==7) & (median_half_year_ten_data['class_']==1)]
# d = median_half_year_ten_data[(median_half_year_ten_data['lag']==10) & (median_half_year_ten_data['class_']==1)]
# =============================================================================

# =============================================================================
# # 对不同class的收益做移动平均，并做图观察
# # 只对一天的累计收益作图
# median_ten_data_ma = median_ten_data.loc[(median_ten_data.lag==3)].copy()
# median_ten_data_ma = median_ten_data_ma[['class_','excess_r_1','date']]
# median_ten_data_ma['ma'] = median_ten_data_ma.groupby('class_')['excess_r_1'].transform(lambda d : d.rolling(30).mean())
# a = median_ten_data_ma[median_ten_data_ma.class_==1]
# fig=plt.figure(figsize=(24,12))
# values = median_ten_data_ma.loc[median_ten_data_ma['class_']==1, 'ma']
# plt.plot(dates,values,linestyle='-',marker='*',c='r',alpha=0.5)
# values = median_ten_data_ma.loc[median_ten_data_ma['class_']==5, 'ma']
# plt.plot(dates,values,linestyle='-',marker='*',c='blue',alpha=0.5)
# values = median_ten_data_ma.loc[median_ten_data_ma['class_']==7, 'ma']
# plt.plot(dates,values,linestyle='-',marker='*',c='blue',alpha=0.5)
# values = median_ten_data_ma.loc[median_ten_data_ma['class_']==9, 'ma']
# plt.plot(dates,values,linestyle='-',marker='*',c='green',alpha=0.5)
# ax=plt.gca()
# date_format=mpl.dates.DateFormatter('%Y-%m-%d')
# ax.xaxis.set_major_formatter(date_format)
# ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(90))#设定坐标轴的显示的刻度间隔
# fig.autofmt_xdate()#防止x轴上的数据重叠，自动调整。
# plt.savefig("F:\\Intern\\3outputFile\\Figure\\median_ma_class_lag3_r1.png")
# 
# # return5和return10有异常值,都不对，太小了
# median_ten_data_ma = median_ten_data.loc[(median_ten_data.lag==3)].copy()
# median_ten_data_ma = median_ten_data_ma[['class_','excess_r_3','date']]
# median_ten_data_ma['ma'] = median_ten_data_ma.groupby('class_')['excess_r_3'].transform(lambda d : d.rolling(30).mean())
# a = median_ten_data_ma[median_ten_data_ma.class_==1]
# fig=plt.figure(figsize=(24,12))
# values = median_ten_data_ma.loc[median_ten_data_ma['class_']==1, 'ma']
# plt.plot(dates,values,linestyle='-',marker='*',c='r',alpha=0.5)
# values = median_ten_data_ma.loc[median_ten_data_ma['class_']==5, 'ma']
# plt.plot(dates,values,linestyle='-',marker='*',c='blue',alpha=0.5)
# ax=plt.gca()
# date_format=mpl.dates.DateFormatter('%Y-%m-%d')
# ax.xaxis.set_major_formatter(date_format)
# ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(90))#设定坐标轴的显示的刻度间隔
# fig.autofmt_xdate()#防止x轴上的数据重叠，自动调整。
# plt.savefig("F:\\Intern\\3outputFile\\Figure\\median_ma_class_lag3_r3.png")
# 
# 
# a = median_ten_data_ma[median_ten_data_ma['excess_r_10']>0.5]
# a = median_ten_data_ma[median_ten_data_ma['ma']>0.5]
# a = median_ten_data[['class_','excess_r_1','lag']]
# a['excess_r_1'] = a['excess_r_1'].astype(float)
# 
# b = a.groupby(['lag','class_'])['excess_r_1'].mean()
# =============================================================================


# 计算扣除交易费用，rankic无影响，不同class下的平均收益减去交易费用即可
fee = 0.003


# 分月研究rankic
rankic_n_data['mon'] = rankic_n_data['date'].apply(lambda s:s[4:6])
rank_er_col = ['rankic_'+ s for s in er_col]
ic_mean_mon = rankic_n_data[['lag','mon']+rank_er_col].groupby(['lag','mon']).agg(np.nanmean)
ic_std_mon = rankic_n_data[['lag','mon']+rank_er_col].groupby(['lag','mon']).agg(np.std)
ic_ir_mon = ic_mean_mon/ic_std_mon*np.sqrt(250)
ic_ir_mon = ic_ir_mon.reset_index()
ic_mean_mon = rankic_n_data.reset_index()



# rankic 在eps为正时，为负，找几天作图观察一下
#  20100311
testdate = '20100311'
test = pd.read_csv("F:\\Intern\\3outputFile\\EPS_excessR\\excess_return\\Factor_Exposure_eps_lag3"+'\\data_%s.csv'%testdate)
test = test[test.EPS_change>0]
import matplotlib.pyplot as plt
%matplotlib inline
plt.scatter(x = test['EPS_change'], y = test['excess_r_10'], s=10)
