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

def load_submit_file(path):
	reader=pd.read_csv(path,iterator=False,delimiter=',',encoding='utf-8')
	r=reader.sort_values(by="score")
	#print r
	uids=np.array(r['uid'])
	return uids

def rank_dict(output1):
	output_dict1={}
	for i in range(len(output1)):
		output_dict1[str(output1[i])]=i
	return output_dict1

def score_dict(path):
	reader=pd.read_csv(path,iterator=False,delimiter=',',encoding='utf-8')
	output_dict1={}
	for i in range(len(reader['uid'])):
		output_dict1[str(reader['uid'][i])]=reader['score'][i]
	return output_dict1


def output_blend(d,path):
	f=open(path,'wb')
	f.write('uid,score\n')
	for uid,score in d.items():
		f.write(str(uid)+','+str(score)+'\n')
	f.close()

def analysis():
	config_instance=config.Config('log_move')
	output1=load_submit_file(config_instance.path_predict+'output/'+"1-15-3-blend.csv")#
	level='level_two'
	name='log_move_xgb2500'
	output1=load_submit_file(config_instance.path_predict+'output/'+level+'_'+name+'.csv')
	#output2=load_submit_file(config_instance.path_predict+level+'/'+name+'.csv')
	output2=load_submit_file(config_instance.path_predict+'output/'+"best.csv")#origin_xgb2500_log
	dict1=rank_dict(output1)
	dict2=rank_dict(output2)

	score_dict1=score_dict(config_instance.path_predict+'output/'+"best.csv")
	score_dict2=score_dict(config_instance.path_predict+'output/'+level+'_'+name+'.csv')
	blend_dict={}
	for uid,score in score_dict1.items():
		blend_dict[uid]=score+score_dict2[uid]

	#output_blend(blend_dict,config_instance.path_predict+'output/'+"blend.csv")
	rank_change=[]
	uids=sorted(output1)
	for uid in uids:
		rank1=dict1.get(str(uid))
		rank2=dict2.get(str(uid))
		rank_change.append(int(rank1)-int(rank2))
		if rank2<100:
			print "uid:",uid,"  rank1:",rank1,"  rank2:",rank2,"  diff:",(rank1-rank2)

	print "var:",np.var(rank_change)," max:",np.max(rank_change)," min:",np.min(rank_change)
	pass


def transform_predict(config,level,name):
	reader=pd.read_csv(config.path_predict+level+'/'+name+'.csv',iterator=False,delimiter=',',encoding='utf-8')
	l=[]
	for i in range(len(reader['score'])):
		l.append([reader['uid'][i],1-reader['score'][i]])

	f=open(config.path_predict+'output/'+level+'_'+name+'.csv','wb')
	f.write('"uid","score"\n')
	for row in l:
		f.write(str(row[0])+','+str(row[1])+'\n')

def main():
	config_instance=config.Config('log_move')
	#transform_predict(config_instance,'level_two','log_move_xgb2500')
	analysis()
	pass

if __name__ == '__main__':
	main()