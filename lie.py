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

def lie():
	config_instance=config.Config('log_move')
	column_dict=score_dict(config_instance.path_predict+'output/'+"best_0.7291.csv")
	column_dict2=sorted(column_dict.items(),key=lambda d:d[1],reverse=True)
	#print column_dict2

	column_dict_new=score_dict(config_instance.path_predict+'output/'+"origin_lr.csv")
	old_ranks=level_ranks(column_dict)
	new_ranks=level_ranks(column_dict_new)

	i=0
	print len(column_dict2)
	aa=0
	#return
	for uid, score in column_dict2:
		# if uid=='19343':
		# 	column_dict[uid]=0
		# elif uid=='3399':
		# 	column_dict[uid]=1

		# if i>2000 and i <2100:
		# 	if new_ranks[uid][0]-old_ranks[uid][0]>1000:
		# 		column_dict[uid]=1

		if i<4100:
			if i==13:
				#column_dict[uid]=0
				print 'bingo:',uid
			i+=1
			continue
		# if uid=='2662' or uid=='2963': #3399=1 19343=0
		# 	print 'bingo'
		# 	column_dict[uid]=1
		print uid,' ',score,' ',old_ranks[uid],' ',new_ranks[uid],' ',column_dict_new[uid]
		# if i%2==1:
		# 	column_dict[uid]=0
		#print i
		
		if new_ranks[uid][0]-old_ranks[uid][0]>1500:
			column_dict[uid]=1
			aa+=1
		i+=1
		
		if i>4200:
			break

	print aa

	output_blend(column_dict,config_instance.path_predict+'output/'+"mix_4100_4200_1500.csv")



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
	lie()
	pass

if __name__ == '__main__':
	main()