# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 10:10:29 2018

@author: sy
"""

import pandas as pd
import numpy as np
import os
import datetime as dt
import pymssql

# 股票所在行业
ind_path = "Z:\\UserData\\industry_daily"
# 股票历史价格
stoc_price_path = "Z:\\UserData\\JYDB\\price_since2005.csv"
# 在指数中的权重
bmkW_ic_path = "Z:\\UserData\\Benchmark\\IC_daily"
bmkW_if_path = "Z:\\UserData\\Benchmark\\IF_daily"
bmkW_ih_path = "Z:\\UserData\\Benchmark\\IH_daily"
# 从券商中取，每天有一个weight
portW_ic_path = "Z:\\PortfolioMonitor\\Backup\\PortWeight\\IC"
portW_if_path = "Z:\\PortfolioMonitor\\Backup\\PortWeight\\IF"
# 自己计算的，实际上每天存储的是根据前一天信息计算的Alpha
alpha_ic_path = "Z:\\PortfolioMonitor\\Alpha\\IC"
alpha_if_path = "Z:\\PortfolioMonitor\\Alpha\\IF"
# 文件夹主要包含哪一天reblance的信息
alphaReb_ic_path = "Z:\\PortfolioMonitor\\RebPortfolio\\IC"
alphaReb_if_path = "Z:\\PortfolioMonitor\\RebPortfolio\\IF"
# 输出路径
out_path = "F:\\Intern\\3outputFile\\Monitor"
out_daily_path = "F:\\Intern\\3outputFile\\Monitor_daily"
def get_DB_t_days(start, end, conn):
    """
    This function will fetch all trading days from db 
    """
    query = """
        SELECT TradingDate
        FROM JYDB..QT_TradingDayNew
        WHERE TradingDate >= '%s' AND TradingDate <= '%s' AND IfTradingDay = 1 AND SecuMarket=83
    """ % (start, end)

    t_days = pd.read_sql(query, conn)
    t_days = t_days["TradingDate"].apply(lambda x:dt.datetime.strftime(x,'%Y%m%d'))
    return list(sorted(np.array(t_days)))

def std_ind_ret(data):
    
    data['Bmk_Weight'] = data['Bmk_Weight']/np.sum(data['Bmk_Weight'])
    
    return np.sum(data['Bmk_Weight'] * data['Ret_Since_PrevReb'])

# 方案一：通过数据库获取交易日
conn =pymssql.connect(server='172.16.37.140', user='jydb', password='jydb', port=1433, database='JYDB')
today = dt.datetime.now()
start_date = today - dt.timedelta(days=365)
trading_day = get_DB_t_days(start_date.strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d'), conn)
prev_day = trading_day[-2]

# 方案二：通过IF_daily/IC_daily文件夹中的文件获取交易日，但是不能取到今天的日期，而且可能有更新不及时的情况
trading_day = sorted([fn.split('.')[1] for fn in os.listdir(bmkW_ic_path)])
prev_day = trading_day[-1]
today = dt.datetime.now().strftime('%Y%m%d')

# stock level

# 测试设置prev_day = '20180201'
# port_if = pd.read_excel(portW_if_path+'\\IF_%s.csv'%prev_day)
prev_day = '20180201'
# SecuCode、 Weight_PrevD
port_if = pd.read_excel("Z:\\PortfolioMonitor\\Sample\\IF_20180201_sample.xlsx", 
                        sheet_name='TargetPort').rename(columns={'Asset ID':'SecuCode', 'Weight':'Weight_PrevD'})

# industry
stoc_ind = pd.read_csv(ind_path+'\\XSIndustry_'+ prev_day +'.csv').rename(columns={'!ID':'SecuCode'})[['SecuCode','XSIndustry']]
port_if = port_if.merge(stoc_ind, on='SecuCode')

# Bmk_Weight\Active_Weight,
bmkW_if = pd.read_table(bmkW_if_path+'\\IF_daily.%s'%prev_day, skiprows=[0,1,2,3,4], delimiter=' ',
                             names = ['SecuCode', 'Type', 'Bmk_Weight'])[['SecuCode','Bmk_Weight']]
bmkW_ic = pd.read_table(bmkW_ic_path+'\\IC_daily.%s'%prev_day, skiprows=[0,1,2,3,4], delimiter=' ',
                             names = ['SecuCode', 'Type', 'Bmk_Weight'])[['SecuCode','Bmk_Weight']]
bmkW_ih = pd.read_table(bmkW_ih_path+'\\IH_daily.%s'%prev_day, skiprows=[0,1,2,3,4], delimiter=' ',
                              names = ['SecuCode', 'Type', 'Bmk_Weight'])[['SecuCode','Bmk_Weight']]
port_if = port_if.merge(bmkW_if, on='SecuCode', how='left')
port_if.loc[port_if['Bmk_Weight'].isnull(), 'Bmk_Weight'] = 0
port_if['Active_Weight'] = port_if.Weight_PrevD - port_if.Bmk_Weight

# Index_Membership(from out file)\Current_Alpha
alpha = pd.read_csv(alpha_if_path+'\\IF_%s.csv'%prev_day).rename(columns={'!ID':'SecuCode','Alpha':'Current_Alpha'})
port_if = port_if.merge(alpha[['SecuCode','Current_Alpha']],on='SecuCode', how='left')
prev_day_if_list = list(bmkW_if['SecuCode'])
prev_day_ic_list = list(bmkW_ic['SecuCode'])
prev_day_ih_list = list(bmkW_ih['SecuCode'])
port_if['SSE50'] = np.nan
port_if['CSI300'] = np.nan
port_if['CSI500'] = np.nan
port_if['SSE50'] = port_if['SecuCode'].apply(lambda s: np.where(s in prev_day_ih_list, 1, 0))
port_if['CSI300'] = port_if['SecuCode'].apply(lambda s: np.where(s in prev_day_if_list, 1, 0))
port_if['CSI500'] = port_if['SecuCode'].apply(lambda s: np.where(s in prev_day_ic_list, 1, 0))


# PrevReb_Alpha\Change_in_Alpha
# 测试设置上一次rebalance日期是20180105
# RebDate_list = sorted([fn[3:11] for fn in os.listdir(alphaReb_if_path)])
# RebDate = RebDate_list[-1]
# Reb_day = trading_day[trading_day.index(RebDate)-1]
Reb_day = '20180104'
alpha_PrevReb = pd.read_csv(alpha_if_path+'\\IF_%s.csv'%Reb_day).rename(columns={'!ID':'SecuCode','Alpha':'PrevReb_Alpha'})
port_if = port_if.merge(alpha_PrevReb[['SecuCode','PrevReb_Alpha']],on='SecuCode', how='left')
port_if['Change_in_Alpha'] = port_if['Current_Alpha'] - port_if['PrevReb_Alpha']

# Ret_Since_PrevReb:不能跨跨区间计算收益率，要计算每一天的收益率累乘，对注释区间做修改
stoc_price = pd.read_csv(stoc_price_path)
# =============================================================================
# port_if = port_if.merge(stoc_price[stoc_price.TradingDay==int(Reb_day)][['SecuCode','ClosePrice']], 
#                         on = 'SecuCode', how='left').rename(columns={'ClosePrice':'PrevRebCP'})
# port_if = port_if.merge(stoc_price[stoc_price.TradingDay==int(prev_day)][['SecuCode','ClosePrice']], 
#                         on = 'SecuCode', how='left').rename(columns={'ClosePrice':'PrevDCP'})
# port_if['Ret_Since_PrevReb'] = (port_if['PrevDCP']-port_if['PrevRebCP'])/port_if['PrevRebCP']
# =============================================================================
port_if['Ret_Since_PrevReb'] = 1
for loopday in trading_day[trading_day.index(Reb_day)+1:trading_day.index(prev_day)+1]:
    port_if = port_if.merge(stoc_price[stoc_price.TradingDay==int(loopday)][['SecuCode','PrevClosePrice','ClosePrice']],
                                       on='SecuCode', how='left')
    port_if['Ret_Since_PrevReb'] = port_if['Ret_Since_PrevReb']*(port_if['ClosePrice']/port_if['PrevClosePrice'])
    port_if.drop(columns=['PrevClosePrice','ClosePrice'], inplace=True)
port_if['Ret_Since_PrevReb'] = port_if['Ret_Since_PrevReb'] - 1

# Industry_Ret_S_PrevReb\Active_Ret_S_PrevReb
# 需要用IF中三百只股票属于同一行业的，
# 实际上要用区间头一天的股票指数权重，所以在bmkW_if_PrevReb表中加上行业和收益计算
bmkW_if_PrevReb = pd.read_table(bmkW_if_path+'\\IF_daily.%s'%Reb_day, skiprows=[0,1,2,3,4], delimiter=' ',
                             names = ['SecuCode', 'Type', 'Bmk_Weight'])[['SecuCode','Bmk_Weight']]
bmkW_if_PrevReb = bmkW_if_PrevReb.merge(stoc_ind,how='left')
# =============================================================================
# # bmkW_if = bmkW_if.merge(stoc_price[stoc_price.TradingDay==int(Reb_day)][['SecuCode','ClosePrice']],
# #                                    on = 'SecuCode', how='left').rename(columns={'ClosePrice':'PrevRebCP'})
# # bmkW_if = bmkW_if.merge(stoc_price[stoc_price.TradingDay==int(prev_day)][['SecuCode','ClosePrice']],
# #                                    on = 'SecuCode', how='left').rename(columns={'ClosePrice':'PrevDCP'})
# bmkW_if['Ret_Since_PrevReb'] = (bmkW_if['PrevDCP']-bmkW_if['PrevRebCP'])/bmkW_if['PrevRebCP']
# =============================================================================
bmkW_if_PrevReb['Ret_Since_PrevReb'] = 1
for loopday in trading_day[trading_day.index(Reb_day)+1:trading_day.index(prev_day)+1]:
    bmkW_if_PrevReb = bmkW_if_PrevReb.merge(stoc_price[stoc_price.TradingDay==int(loopday)][['SecuCode','PrevClosePrice','ClosePrice']],
                                       on='SecuCode', how='left')
    bmkW_if_PrevReb['Ret_Since_PrevReb'] = bmkW_if_PrevReb['Ret_Since_PrevReb']*(bmkW_if_PrevReb['ClosePrice']/bmkW_if_PrevReb['PrevClosePrice'])
    bmkW_if_PrevReb.drop(columns=['PrevClosePrice','ClosePrice'], inplace=True)
bmkW_if_PrevReb['Ret_Since_PrevReb'] = bmkW_if_PrevReb['Ret_Since_PrevReb'] - 1

bmkW_if_ind = bmkW_if_PrevReb.groupby('XSIndustry').apply(lambda d: std_ind_ret(d)).reset_index().rename(columns={0:'Industry_Ret_S_PrevReb'})
port_if = port_if.merge(bmkW_if_ind, how='left')
port_if['Active_Ret_S_PrevReb'] = port_if['Ret_Since_PrevReb'] - port_if['Industry_Ret_S_PrevReb']

# 调整显示顺序
col_list = ['SecuCode','XSIndustry','SSE50','CSI300','CSI500','Weight_PrevD','Bmk_Weight',
            'Active_Weight','PrevReb_Alpha','Current_Alpha','Change_in_Alpha','Ret_Since_PrevReb',
            'Industry_Ret_S_PrevReb','Active_Ret_S_PrevReb']
port_if = port_if[col_list]


# portfolio level
# Return 需要算每一天的股票收益，加权(权重每天都不一样)得到组合收益和bmk(IF)收益
if_Ret_S_PrevReb = pd.DataFrame()
for loopdate in trading_day[trading_day.index(Reb_day)+1:trading_day.index(prev_day)+1]:
    # portfolio
    loop_port_if = pd.read_excel("Z:\\PortfolioMonitor\\Sample\\IF_20180201_sample.xlsx", 
                        sheet_name='TargetPort').rename(columns={'Asset ID':'SecuCode'})[['SecuCode','Weight']]
    
    loop_port_if = loop_port_if.merge(stoc_price[stoc_price.TradingDay==int(loopdate)][['SecuCode','PrevClosePrice','ClosePrice']],
                         on='SecuCode', how='left')
    loop_port_if['Ret'] = (loop_port_if['ClosePrice']-loop_port_if['PrevClosePrice'])/loop_port_if['PrevClosePrice']
    if_Ret_S_PrevReb.loc[loopdate, 'Portfolio_Ret'] = np.sum(loop_port_if['Weight']*loop_port_if['Ret'])
    # benchmark
    loop_bmk_if = pd.read_table(bmkW_if_path+'\\IF_daily.%s'%loopdate, skiprows=[0,1,2,3,4], delimiter=' ',
                             names = ['SecuCode', 'Type', 'Weight'])[['SecuCode','Weight']]
    loop_bmk_if = loop_bmk_if.merge(stoc_price[stoc_price.TradingDay==int(loopdate)][['SecuCode','PrevClosePrice','ClosePrice']],
                         on='SecuCode', how='left')
    loop_bmk_if['Ret'] = (loop_bmk_if['ClosePrice']-loop_bmk_if['PrevClosePrice'])/loop_bmk_if['PrevClosePrice']
    if_Ret_S_PrevReb.loc[loopdate, 'Bmk_Ret'] = np.sum(loop_bmk_if['Weight']*loop_bmk_if['Ret'])
if_Ret_S_PrevReb['Active_Ret'] = if_Ret_S_PrevReb['Portfolio_Ret'] - if_Ret_S_PrevReb['Bmk_Ret']
if_Ret_S_PrevReb.reset_index(inplace=True)
# Industry
if_industry_expo = port_if.groupby('XSIndustry')[['Weight_PrevD','Bmk_Weight']].agg(sum).rename(columns={'Weight_PrevD':'Portfolio_Weight'})
if_industry_expo['Active_Weight'] = if_industry_expo['Portfolio_Weight'] - if_industry_expo['Bmk_Weight']
# Index
if_index_expo = pd.DataFrame()
if_index_expo.loc['SSE50','Portfolio_Weight']= sum(port_if[port_if['SSE50']==1]['Weight_PrevD'])
if_index_expo.loc['CSI300','Portfolio_Weight']= sum(port_if[port_if['CSI300']==1]['Weight_PrevD'])
if_index_expo.loc['CSI500','Portfolio_Weight']= sum(port_if[port_if['CSI500']==1]['Weight_PrevD'])
if_index_expo.loc['SSE50','Bmk_Weight']= sum(port_if[port_if['SSE50']==1]['Bmk_Weight'])
if_index_expo.loc['CSI300','Bmk_Weight']= sum(port_if[port_if['CSI300']==1]['Bmk_Weight'])
if_index_expo.loc['CSI500','Bmk_Weight']= sum(port_if[port_if['CSI500']==1]['Bmk_Weight'])
if_index_expo['Active_Weight'] = if_index_expo['Portfolio_Weight'] - if_index_expo['Bmk_Weight']
if_index_expo.reset_index(inplace=True)
# 用以计算实时收益的IF股票行业及权重也要输出


# 生成测试用的临时的weight表
# =============================================================================
# temp = port_if[['SecuCode','Weight_PrevD']].rename(columns={'Weight_PrevD':'Weight'})
# for day in trading_day[trading_day.index(Reb_day)+1:trading_day.index(prev_day)+1]:
#     tempdf = temp.copy()
#     tempdf['Weight'] = tempdf['Weight']*(1+np.random.randn(tempdf['Weight'].shape[0]))
#     tempW = pd.ExcelWriter(portW_if_path+'\\IF_%s.xlsx'%day)
#     tempdf.to_excel(tempW, sheet_name='TargetPort')
#     tempW.save()
# =============================================================================

# 将数据中的小数以保留两位的百分数显示
# =============================================================================
# decimal_col = ['Weight_PrevD', 'Bmk_Weight', 'Active_Weight', 
#               'Ret_Since_PrevReb', 'Industry_Ret_S_PrevReb',
#                'Active_Ret_S_PrevReb']
# =============================================================================


# 写入到daily文件夹下的excel中
writer = pd.ExcelWriter(out_daily_path+'\\Monitor_%s.xlsx'%today)
port_if.to_excel(writer, sheet_name='Portfolio_IF', index=False)
if_Ret_S_PrevReb.to_excel(writer, sheet_name='Return_IF')
if_industry_expo.to_excel(writer, sheet_name='Industry_Exposure_IF')
if_index_expo.to_excel(writer, sheet_name='Index_Exposure_IF')
bmkW_if[['SecuCode','Bmk_Weight','XSIndustry']].to_excel(writer, sheet_name='IF_Industry_Weight', index=False)
writer.save()
# 写入到非daily文件夹下的excel中
writer = pd.ExcelWriter(out_path+'\\Monitor_Base.xlsx')
port_if.to_excel(writer, sheet_name='Portfolio_IF', index=False)
if_Ret_S_PrevReb.to_excel(writer, sheet_name='Return_IF')
if_industry_expo.to_excel(writer, sheet_name='Industry_Exposure_IF')
if_index_expo.to_excel(writer, sheet_name='Index_Exposure_IF')
bmkW_if[['SecuCode','Bmk_Weight','XSIndustry']].to_excel(writer, sheet_name='IF_Industry_Weight', index=False)
writer.save()












