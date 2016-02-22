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
	column_dict=score_dict(config_instance.path_predict+'output/'+"best.csv")
	column_dict2=sorted(column_dict.items(),key=lambda d:d[1],reverse=True)
	#print column_dict2
	i=0

	for uid, score in column_dict2:
		if i<61:
			i+=1
			continue
		print score
		column_dict[uid]=1
		if i%2==1:
			column_dict[uid]=0
		i+=1
		if i>75:
			break

	output_blend(column_dict,config_instance.path_predict+'output/'+"blend_lie_5_1.csv")



def main():
	config_instance=config.Config('log_move')
	lie()
	pass

if __name__ == '__main__':
	main()