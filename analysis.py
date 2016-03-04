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

"""
analysis.py
结果分析工具函数库
load_submit_file: 简单读取提交结果
rank_dict: 提交结果中各个uid的排名
score_dict: 不同提交的uid的分数
output_blend: 输出blend结果
analysis: 分析函数，将新预测结果与已有的最好结果进行排名比较，如果偏差过大，则放弃提交以节省线上提交次数
transform_predict: 转换函数，将输出结果作1-predict处理，因为前面说到XGBoost对负类标记为1的预测效果更好，则提交时要重新将负类置为0
"""

def load_submit_file(path):
	reader=pd.read_csv(path,iterator=False,delimiter=',',encoding='utf-8')
	r=reader.sort_values(by="score")
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
	f.write('"uid","score"\n')
	for uid,score in d.items():
		f.write(str(uid)+','+str(score)+'\n')
	f.close()

def analysis():
	config_instance=config.Config('log_move')
	output1=load_submit_file(config_instance.path_predict+'output/'+"1-15-3-blend.csv")#
	level='level_one'
	name='log_move_xgb2500'
	output2=load_submit_file(config_instance.path_predict+'output/'+level+'_'+name+'.csv')
	dict1=rank_dict(output1)
	dict2=rank_dict(output2)

	#rank change
	rank_change=[]
	uids=sorted(output1)
	for uid in uids:
		rank1=dict1.get(str(uid))
		rank2=dict2.get(str(uid))
		rank_change.append(int(rank1)-int(rank2))
	print "var:",np.var(rank_change)," max:",np.max(rank_change)," min:",np.min(rank_change)

	#blend
	score_dict1=score_dict(config_instance.path_predict+'output/'+"best.csv")
	level='level_two'
	name='log_move_lr_sag'
	score_dict2=score_dict(config_instance.path_predict+'output/'+level+'_'+name+'.csv')
	max_1=max(score_dict1.values())
	max_2=max(score_dict2.values())

	min_1=min(score_dict1.values())
	min_2=min(score_dict2.values())

	print max_1
	print max_2
	print min_1
	print min_2
	blend_dict={}
	for uid,score in score_dict1.items():
		if score==1 or score==0:
			blend_dict[uid]=score
		else:
			blend_dict[uid]=(0.73*8*(score-min_1)/(max_1-min_1)+0.71*(score_dict2[uid]-min_2)/(max_2-min_2))/(0.73*8+0.71)

	output_blend(blend_dict,config_instance.path_predict+'output/'+"blend_8.csv")

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
	transform_predict(config_instance,'level_one','log_move_rf1000')
	#analysis()
	pass

if __name__ == '__main__':
	main()