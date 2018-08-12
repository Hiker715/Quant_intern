#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 08:57:18 2018

@author: heisenberg
"""

import os
import pandas as pd
import numpy as np
from dateutil.parser import parse
import datetime as dt
import pymssql

# IF_daily should be used so this function is ussless
def ic_if_fn(date_):
    
    """
    return corresponding IC or IF filename of specific date 
    """
    m_dict = {'10':'A','11':'B','12':'C'}
    date_ = date_.strftime("%Y%m%d")
    y = date_[2:4]
    m = date_[4:6]
    if m[0] == '0':
        m = m[1]
    else:
        m = m_dict[m]
        
    return y+m



def standerize_return(data, return_l):
    
    data['Weight'] = data['Weight']/np.sum(data['Weight'])
    ind_return_l = []
    for r in return_l:
        ind_return_l.append(np.sum(data['Weight'] * data[r]))
    return pd.Series(ind_return_l)

def fetch_full_dates(dates_path):
    
    """
    This function will return a list of all days(as str) for compute factor, as string
    """
    
    fns= os.listdir(dates_path)
    dates = [fn.split('.')[1] for fn in fns]
    return dates

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


def cal_excess_return(fulldates, eps_lag_list = [3,5,7,10,15,20], lag=[1,3,5,10,15,20,25,30], reset=0):
    
    """
    return excess return of fulldates
    lag:{1:"one day return", n:"n day return"}
    """

    stock_p = pd.read_csv(path_stock_p)

    stock_p['day_return'] = stock_p.ClosePrice/stock_p.PrevClosePrice
    for l in lag:
    	stock_p['return_%s'%l] = 1
    	for i in range(l):
    		stock_p['return_%s'%l] = stock_p['return_%s'%l] * stock_p.groupby('SecuCode')['day_return'].transform(lambda x:x.shift(-i))
    	stock_p['return_%s'%l] = stock_p['return_%s'%l] - 1
    
    stock_p[stock_p == np.inf] = 0
    for eps_lag in eps_lag_list:
	    factor_name = "excess_return"
	    _path_factor = "\\".join([path_out, factor_name])
	    path_factor = "\\".join([_path_factor, 'Factor_Exposure_eps_lag%s'%eps_lag])

	    # compute industry return
	    

	    # check the existence of the folders
	    if not os.path.exists(_path_factor):
	        os.makedirs(_path_factor)
	    if not os.path.exists(path_factor):
	        os.makedirs(path_factor)
	         
	    # get non exists dates
	    if type(fulldates[0]) == type(" "):
	        fulldates = [parse(x) for x in fulldates]
	    #exist_dates = [parse(i[5:13]) for i in os.listdir(path_factor)]
	    exist_dates = []
	    for i in os.listdir(path_factor):
	    	try:
	    		exist_dates.append(i[5:13])
	    	except:
	    		pass

	    non_exist_dates = [i for i in fulldates if i  not in exist_dates]

	     # run with reset
	    if reset == 1:
	        non_exist_dates = fulldates
	    
	    for num, loopdate in enumerate(non_exist_dates):

	        if num<10 or num%100==0:
	            print("now {0}  lag {1} {2}...".format(factor_name, lag, loopdate))
	        
	        loop_indu = pd.read_csv(path_indu+'\\XSIndustry_' + loopdate.strftime('%Y%m%d') + '.csv', skiprows=[0],
	                                   header=None, names=['ID','Type','Name','XSIndustry'])

	        loop_univ = pd.read_table(path_univ+'\\univ_all_A_daily.' + loopdate.strftime('%Y%m%d'), skiprows=[0,1,2,3,4],
	                            header=None, names=['SecuCode', 'Type', 'Num'], delimiter=" ")
	        loop_univ = loop_univ[['SecuCode']].merge(loop_indu[['ID','XSIndustry']],
	                     left_on = 'SecuCode', right_on = 'ID', how = 'left')[['SecuCode','XSIndustry']]        
	        loop_eps = pd.read_csv(path_eps + '\\Factor_Exposure_lag_%s\\data_'%eps_lag + loopdate.strftime('%Y%m%d') + '.csv')
	        # compute different lag excess return
	        return_list = []
	        for l in lag:
	            return_list.append('return_%s'%l)
	        stock_p_col_list = ['SecuCode'] + return_list
	         # combine stockcode and industry and return
	        loop_univ = loop_univ.merge(stock_p[stock_p.TradingDay==int(loopdate.strftime('%Y%m%d'))][stock_p_col_list], how='left')

	        # fetch stock weights, we should use last tradingday's weight to compute today's industry return 
	        loopdate_prev = tradingday[np.argwhere(tradingday==loopdate.strftime('%Y%m%d')) - 1][0][0]
	        
	        ic_w = pd.read_table(path_ic + '\\IC_daily.' + loopdate_prev, skiprows=[0,1,2,3,4], header=None,
	                            names=['SecuCode','Type', 'Weight'], delimiter=" ")
	        if_w = pd.read_table(path_if + '\\IF_daily.' + loopdate_prev, skiprows=[0,1,2,3,4], header=None,
	                            names=['SecuCode','Type', 'Weight'], delimiter=" ")

	        # fetch IC's or IF's stock industry and return
	        ic_w = ic_w.merge(loop_univ, on='SecuCode', how='left')
	        if_w = if_w.merge(loop_univ, on='SecuCode', how='left')

	        ic_w = ic_w.merge(ic_w.groupby(['XSIndustry'])[['Weight'] + return_list].apply(lambda x: standerize_return(x, return_list)).reset_index(), how='left').rename(columns={0:'ind_return_1',1:'ind_return_3',2:'ind_return_5',3:'ind_return_10',4:'ind_return_15',5:'ind_return_20',6:'ind_return_25',7:'ind_return_30'})
	        if_w = if_w.merge(if_w.groupby(['XSIndustry'])[['Weight'] + return_list].apply(lambda x: standerize_return(x, return_list)).reset_index(), how='left').rename(columns={0:'ind_return_1',1:'ind_return_3',2:'ind_return_5',3:'ind_return_10',4:'ind_return_15',5:'ind_return_20',6:'ind_return_25',7:'ind_return_30'})


	        ind_return_list = []
	        for l in lag:
	            ind_return_list.append('ind_return_%s'%l)
	            
	        loop_univ = loop_univ.merge(if_w[['SecuCode'] + ind_return_list], how='left').set_index('XSIndustry')

	        loop_univ.update(ic_w[['XSIndustry'] + ind_return_list].groupby('XSIndustry').agg(np.mean),overwrite=False)
	        loop_univ.update(if_w[['XSIndustry'] + ind_return_list].groupby('XSIndustry').agg(np.mean),overwrite=False)
	        loop_univ.reset_index(inplace=True)

	        # some ind_return is na, check for special cases
	        loop_univ = loop_univ[~loop_univ.XSIndustry.isnull()]


	        for l in lag:
	            loop_univ['excess_r_%s'%l] = loop_univ['return_%s'%l] -  loop_univ['ind_return_%s'%l] 
	        loop_univ = loop_eps.merge(loop_univ, on='SecuCode', how='left')
	        if loop_univ.isnull().values.any():
	            with open(log_path + "\\naFile.txt",'w') as f:
	                f.write(path_factor + "   data_%s.csv"%loopdate.strftime("%Y%m%d")+'  has na values')
	        loop_univ.to_csv(path_factor + '\\data_' + loopdate.strftime("%Y%m%d") + ".csv", index=False)

    

if __name__ == '__main__':
    # set the environment
    path_indu = "Z:\\UserData\\industry_daily"
    path_univ = "Z:\\UserData\\universe\\univ_all_A_daily"
    path_ic = "Z:\\UserData\\Benchmark\\IC_daily"
    path_if = "Z:\\UserData\\Benchmark\\IF_daily"
    path_stock_p = "Z:\\UserData\\JYDB\\price_since2005.csv"
    path_eps = "F:\\Intern\\3outputFile\\EPSFile\\EPSAvg_change_perc"
    path_out = "F:\\Intern\\3outputFile\\EPS_excessR"
    log_path = "F:\\Intern\\3outputFile\\outTXT"
    # fulldates is the days we want to reasearch on, tradingday is just tradingday
    conn =pymssql.connect(server='172.16.37.140', user='jydb', password='jydb', port=1433, database='JYDB')
    tradingday = get_DB_t_days('20100129', '20180731', conn)
    fulldates = tradingday[1:]
    cal_excess_return(fulldates, lag=[1,3,5,10,15,20,25,30], reset=0)

