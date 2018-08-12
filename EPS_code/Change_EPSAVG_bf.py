
import pandas as pd
import os
import numpy as np


def push_eps(pather, path_out):


	for fn in os.listdir(pather):
		loopdate = pd.read_csv(pather+"\\"+fn)
		# if EPS_change is zero, do not need to change EPSAvg_bf
		loopdate.loc[loopdate['EPS_change']!=0,'EPSAvg_bf'] = loopdate.loc[loopdate['EPS_change']!=0,'EPSAvg_bf'].apply(lambda v: max(v,0.02) if v>=0 else min(v, -0.02))
		loopdate['EPS_change'] = (loopdate['EPSAvg'] - loopdate['EPSAvg_bf']) / np.abs(loopdate['EPSAvg_bf'])
		if not os.path.exists(path_out):
			os.makedirs(path_out)
		loopdate.to_csv(path_out+"\\"+fn, index=False)

if __name__== "__main__":

	path_excessER = "F:\\Intern\\3outputFile\\EPS_excessR\\excess_return_origin"
	path_Out = "F:\\Intern\\3outputFile\\EPS_excessR\\excess_return"
	eps_lag = [3,5,7,10,15,20]
	for l in eps_lag:
		pather = path_excessER + "\\Factor_Exposure_eps_lag%s"%l
		path_out = path_Out + "\\Factor_Exposure_eps_lag%s"%l
		push_eps(pather, path_out)

