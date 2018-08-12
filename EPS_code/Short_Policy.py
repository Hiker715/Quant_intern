# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 15:19:16 2018

@author: sy
"""
# 另外任务：当epschange为正的时候，收益是不是正的


# =============================================================================
# 策略1.0
# 做空对象：epschange处于半年周期中最大的10%的股票（class_=10）
# 半年周期可以选择固定窗口和滑动窗口
# 操作：对做空对象做空N天，计算所有每天股票的平均超额收益和总超额收益
# N：20，30
# lag：3，5，7，10，15，20
# 收益为当天的（超额）收益，满三十天剔除
# =============================================================================


import pandas as pd
import numpy as np
import pymssql
import datetime as dt
from collections import defaultdict

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


# =============================================================================
# 研究指定区间内的交易日，研究指定收益率
# =============================================================================
conn =pymssql.connect(server='172.16.37.140', user='jydb', password='jydb', port=1433, database='JYDB')
tradingday = get_DB_t_days('20100201', '20180731', conn)
fulldates = tradingday
eps_lag = [3,5,7,10,15,20]
er_col = ['excess_r_1', 'excess_r_3', 'excess_r_5', 'excess_r_10', 'excess_r_15','excess_r_20', 'excess_r_25','excess_r_30']
er_col = er_col[-2:]


# =============================================================================
# 获取满足条件的股票，固定窗口
# =============================================================================
def stock_list_df(fulldates, er_path, eps_lag = eps_lag):
    # 本函数取class_=10的股票，取其他class_的股票只需要修改percentile部分
    short_stc_data = pd.DataFrame()
    for lag in eps_lag:
        pather = path_eR + '\\Factor_Exposure_eps_lag%s'%lag
        stcdf = pd.DataFrame()
        seq = np.linspace(0,len(fulldates),18)
        for i in range(17):
            tempdf = pd.DataFrame()
            for loopdate in fulldates[int(seq[i]):int(seq[i+1])]:
                looper = pd.read_csv(pather + '\\data_%s.csv'%loopdate)
                looper = looper[looper.EPS_change<0]
                tempdf = pd.concat([tempdf, looper])
            cut_perct = np.percentile(tempdf['EPS_change'],10)
            tempdf = tempdf[tempdf.EPS_change<=cut_perct]
            stcdf = pd.concat([stcdf, tempdf])
        stcdf['lag'] = lag
        short_stc_data = pd.concat([short_stc_data, stcdf])
    
    return short_stc_data

short_stc_data = stock_list_df(fulldates, path_eR)      
short_stc_data.to_csv("F:\\Intern\\2CompanyFile\\EPStask\\short_stc_data.csv", index=False)
short_stc_data = pd.read_csv("F:\\Intern\\2CompanyFile\\EPStask\\short_stc_data.csv")
# =============================================================================
# 获取满足条件的股票，滑动窗口
# =============================================================================
def stock_list_df_cycle(fulldates, path_eR, cycle=120, eps_lag = eps_lag):
    # 本函数以cycle为时间窗口周期，取出窗口期下class为10的股票
    short_stc_data = pd.DataFrame()
    fullEndDates = [d[:4]+'-'+d[4:6]+'-'+d[6:] for d in fulldates]
    for lag in eps_lag:
        pather = path_eR + '\\Factor_Exposure_eps_lag%s'%lag
        stcdf = pd.DataFrame()
        tempdf = pd.DataFrame()
        for i, loopdate in enumerate(fulldates):
           
            if i<cycle:
                looper = pd.read_csv(pather + '\\data_%s.csv'%loopdate)
                looper = looper[looper.EPS_change<0]
                tempdf = pd.concat([tempdf, looper])
            else:
                cut_perct = np.percentile(tempdf['EPS_change'],10)
                tempdf = tempdf[tempdf.EndDate!=fullEndDates[i-cycle]]
                looper = pd.read_csv(pather + '\\data_%s.csv'%loopdate)
                looper = looper[looper.EPS_change<0]
                tempdf = pd.concat([tempdf, looper])
                looper = looper[looper.EPS_change<=cut_perct]
                stcdf = pd.concat([stcdf, looper])
        stcdf['lag'] = lag
        short_stc_data = pd.concat([short_stc_data, stcdf])
    
    return short_stc_data

short_stc_data_c = stock_list_df_cycle(fulldates, path_eR)      
short_stc_data_c.to_csv("F:\\Intern\\2CompanyFile\\EPStask\\short_stc_data_cycle.csv", index=False)
short_stc_data_c = pd.read_csv("F:\\Intern\\2CompanyFile\\EPStask\\short_stc_data_cycle.csv")


# =============================================================================
# 计算手中持有的股票
# 当天持有，包括当天的接下来N天不动
# 资金十等分是不是体现在做资本累计收益曲线上
# =============================================================================
def f_hold_stock(stockdata, fulldates, cumday, eps_lagl = eps_lag):
    # columns: date, secucode, lag
    # 结果都是none，检查，因为update返回none
    # 只返回了lag为3的数据，因为在循环中添加进了本来不该被修改但每次循环都被修改的数据:fulldates
    hold_stc = pd.DataFrame()
    fulldates = [d[:4]+'-'+d[4:6]+'-'+d[6:] for d in fulldates]
    for l in eps_lagl:
        temp_df = pd.DataFrame()
        hold_dict = defaultdict(set)
        
        for i, loopdate in enumerate(fulldates):
            if i>=cumday:
                hold_dict[fulldates[i]] = hold_dict[fulldates[i]] - hold_dict[fulldates[i-cumday]]

            temp_stock = set(stockdata.loc[(stockdata.EndDate==loopdate) & (stockdata['lag']==l),'SecuCode'])
            followdate = fulldates[i:i+cumday]

            for date in followdate:
                hold_dict[date].update(temp_stock)
        
        for k,v in hold_dict.items():
            temp = pd.DataFrame()
            temp['SecuCode'] = list(v)
            temp['EndDate'] = k
            temp_df = pd.concat([temp_df, temp])
        
        pather = "F:\\Intern\\3outputFile\\EPS_excessR\\excess_return\\Factor_Exposure_eps_lag%s"%l
        for loopdate in fulldates:
            ldate = loopdate[:4]+loopdate[5:7]+loopdate[8:]
            looper = pd.read_csv(pather+'\\data_%s.csv'%ldate)[['SecuCode','EndDate','excess_r_1']]
            temp_df = temp_df.merge(looper, on = ['EndDate','SecuCode'],how='left')
        temp_df['lag'] = l
        hold_stc = pd.concat([hold_stc, temp_df])
    hold_stc['excess_r_1'] = hold_stc['excess_r_1']*(-1)
    
    return hold_stc


hold_stock = f_hold_stock(short_stc_data, fulldates, 30, eps_lag).rename(columns={}).sort_values(by=['lag','date'])
hold_stock['mon'] = hold_stock['date']
hold_stock_20 = f_hold_stock(short_stc_data, fulldates, 20, eps_lag).sort_values(by=['lag','date'])




hold_stock.to_csv("F:\\Intern\\2CompanyFile\\EPStask\\hold_stock.csv", index=False)
hold_stock = pd.read_csv("F:\\Intern\\2CompanyFile\\EPStask\\hold_stock.csv")

hold_stock_20.to_csv("F:\\Intern\\2CompanyFile\\EPStask\\hold_stock_20.csv", index=False)
hold_stock_20 = pd.read_csv("F:\\Intern\\2CompanyFile\\EPStask\\hold_stock——20.csv")

def p_rate(d):
    return sum(d>0)/len(d)

hold_stock_agg = hold_stock.groupby(['lag','date'])['excess_r_1'].agg([np.mean,sum]).reset_index()
win_rate = hold_stock_agg.groupby('lag')['mean','sum'].agg(p_rate)
hold_stock_agg_20 = hold_stock_20.groupby(['lag','date'])['excess_r_1'].agg([np.mean,sum]).reset_index()
win_rate_20 = hold_stock_agg_20.groupby('lag')['mean','sum'].agg(p_rate)




























