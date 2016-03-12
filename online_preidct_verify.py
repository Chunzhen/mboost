#! /usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import time

import load_data
import config
import matplotlib.pyplot as plt

"""
线上预测集blend
利用Local_predict_verify观测到的blend优化区间结果，进行线上的blend
"""

def score_dict(path):
	reader=pd.read_csv(path,iterator=False,delimiter=',',encoding='utf-8')
	output_dict1={}
	for i in range(len(reader['uid'])):
		output_dict1[str(reader['uid'][i])]=reader['score'][i]
	return output_dict1

def output_blend(d,path):
	f=open(path,'wb')
	f.write('"uid","score"\n')
	for uid,score in d.items():
		f.write(str(uid)+','+str(score)+'\n')
	f.close()

def blend():
	config_instance=config.Config('log_move')
	#当前最好的XGBoost模型最好结果
	column_dict=score_dict(config_instance.path_predict+'output/'+"best_0.7291.csv")
	column_dict2=sorted(column_dict.items(),key=lambda d:d[1],reverse=True)
	#print column_dict2
	#逻辑回归的结果
	column_dict_new=score_dict(config_instance.path_predict+'output/'+"origin_lr.csv")
	old_ranks=level_ranks(column_dict)
	new_ranks=level_ranks(column_dict_new)

	i=0
	print len(column_dict2)
	aa=0
	diffs=[]
	for uid, score in column_dict2:
		if uid=='19343':
			column_dict[uid]=0
		elif uid=='3399':
			column_dict[uid]=1

		if i>=2000 and i <=2100:
			if new_ranks[uid][0]-old_ranks[uid][0]>1000:
				column_dict[uid]=1

		if i>=4600 and i <=4700:
			if new_ranks[uid][0]-old_ranks[uid][0]>1100:
				column_dict[uid]=1

		if i>=4800 and i<=4900:
			if new_ranks[uid][0]-old_ranks[uid][0]>1100:
				column_dict[uid]=1

		if i>=4200 and i<=4300:
			if new_ranks[uid][0]-old_ranks[uid][0]>1200:
				column_dict[uid]=1

		print uid,' ',score,' ',old_ranks[uid],' ',new_ranks[uid],' ',column_dict_new[uid]


		i+=1
	print aa

	output_blend(column_dict,config_instance.path_predict+'output/'+"mix_all_correct.csv")

def print_diff(diffs):
		plt.plot(range(len(diffs)),diffs)
		plt.show()

def level_ranks(column_dict):
		column_dict2=sorted(column_dict.items(),key=lambda d:d[1])
		i=0
		ranks={}
		for uid, score in column_dict2:
			rank=ranks.get(uid,[])
			rank.append(i)
			ranks[uid]=rank
			i+=1

		return ranks

def main():
	config_instance=config.Config('log_move')
	blend()
	pass

if __name__ == '__main__':
	main()