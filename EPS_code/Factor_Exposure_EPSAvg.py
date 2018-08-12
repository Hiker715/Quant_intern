#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 08:57:18 2018

@author: heisenberg
"""

import pandas as pd
import numpy as np
import pymssql
import os
from dateutil.parser import parse
import datetime as dt


def fetch_data(query, db = 'JYDB'):
    """
    This function will fetch data from db according to query

    parameter(s)
    ----------
    db: target database name
    query: SQL Query

    returns
    -------
    the dataframe of the data
    """

    conn = pymssql.connect(host='172.16.37.140',port='1433',user='jydb',password='jydb',database="JYDB",charset="utf8")
    # print("Reading Data from " + db + "...")
    data = pd.read_sql(query, conn)
    # print("Read Data from " + db + "Successfully")
    conn.close()
    return data


def fetch_full_dates(dates_path):
    
    """
    This function will fetch all days for compute factor 
    """
    
    fns= os.listdir(dates_path)
    dates = []
    for fn in fns:
        try:
            dates.append(fn.split('.')[1])
        except:
            print(fn+' file read wrong**************')
    return dates


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
    t_days = t_days["TradingDate"].apply(lambda x:dt.datetime.strftime(x,'%Y-%m-%d'))
    return np.array(t_days)


def EPSAvg_change_perc(fulldates, lag_list = [3,5,7,10,15,20], reset=0):
    
    for lag in lag_list: 
        # set up environment
         factor_name = "EPSAvg_change_perc"
         _path_factor = "\\".join([path_output, factor_name])
         path_factor = "\\".join([_path_factor, 'Factor_Exposure_lag_%s'%lag])
         
         # check the existence of the folders
         if not os.path.exists(_path_factor):
             os.makedirs(_path_factor)
         if not os.path.exists(path_factor):
             os.makedirs(path_factor)
             
        # get non exists dates
         if type(fulldates[0]) == type(" "):
            fulldates = [parse(x) for x in fulldates]
    
         exist_dates = []
         for i in os.listdir(path_factor):
            try:
               exist_dates.append(parse(i[5:13]))
            except:
                pass 
         non_exist_dates = [i for i in fulldates if i  not in exist_dates]
         
         # get lag days trading dates
         t_days_lag = t_days[lag: ]
        
        # run with reset
         if reset == 1:
            non_exist_dates = fulldates
        
        # prepare data
         for num, loopdate in enumerate(non_exist_dates):
            if num <= 10 or num % 100 == 0:
                print("now {0} {1} ...".format(factor_name, loopdate))
            loop_univ = pd.read_table(path_univ + '/univ_all_A_daily.' + loopdate.strftime('%Y%m%d'), skiprows=[0,1,2,3,4],
                                      header=None, names=['SecuCode', 'Type', 'Num'], delimiter=" ")[['SecuCode']].merge(Mapping, how='left')
            
            idx = t_days_lag.index(loopdate)
            loopdate_lag_bf = t_days[idx]
            loop_univ = loop_univ[loop_univ.EndDate==loopdate].merge(loop_univ[loop_univ.EndDate==loopdate_lag_bf], 
                                 on='SecuCode', how='left', suffixes=['','_bf'])
            loop_univ['EPS_change'] = (loop_univ.EPSAvg - loop_univ.EPSAvg_bf) / np.abs(loop_univ.EPSAvg_bf)
            
            loop_univ.to_csv(path_factor+ "\\data_" + loopdate.strftime("%Y%m%d") + ".csv", index=False)
            
    

if __name__ == '__main__':
    path_output = "F:\\Intern\\3outputFile\\EPSFile"
    path_univ = "Z:\\UserData\\universe\\univ_all_A_daily"
    Mapping = fetch_data("""select su.SecuCode, pf.EndDate, pf.EPSAvg from JYDB.dbo.C_EX_ProForStat pf
                                left join JYDB.dbo.SecuMain su on pf.InnerCode=su.InnerCode
                                where pf.EndDate>'2009-12-31' and pf.ForeYearLevel='t' """)
    Mapping.SecuCode = Mapping.SecuCode.apply(int)
    conn =pymssql.connect(server='172.16.37.140', user='jydb', password='jydb', port=1433, database='JYDB')
    t_days = get_DB_t_days('20071231','20180731',conn)
    t_days = pd.DataFrame({"EndDate": [parse(t) for t in t_days]})
    # only take trading days‘data in Mapping
    Mapping = t_days.merge(Mapping, how='left')
    t_days = list(t_days.EndDate)
    fulldates = get_DB_t_days('20100129','20180731',conn)
    EPSAvg_change_perc(fulldates)



