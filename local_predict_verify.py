#! /usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import time

import load_data
from config import Config

from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

class Local_predict_verify(object):
	"""
	:class Local_predict_verify
	:本地验证集结果验证和blend类
	:本类的主要作用是描绘本地验证集中不同分类器排名差的分布，通过分布来确定选择blend区间
	:这是由于XGBoost这样的分类器对0,1两侧的分类能力很强，而对0.5附近的分类能力不强。而全局用其他分类器blend的效果会使0.5附近的分类效果变好，
	:却使0,1两端的分类结果变差，所以要从中间截取一段区域去blend，才会对整体的结果有提升
	"""
	def __init__(self,config,level,clf_name):
		self.config=config
		self.level=level
		self.__clf_name=clf_name

	def load_clf_file(self,level,name):
		reader=pd.read_csv(self.config.path_predict+level+'/'+name+'.csv',iterator=False,delimiter=',',encoding='utf-8')
		d={}
		for i in range(len(reader['uid'])):
			d[reader['uid'][i]]=reader['score'][i]
		return d

	def level_data(self):
		level=self.level
		clf_name=self.__clf_name
		#读取验证集数据
		load_data_instance=load_data.Load_data(self.config)
		X_0,test_X_0,X_1,test_X_1,uid_0,test_uid_0,uid_1,test_uid_1=load_data_instance.train_test_xy(1)

		test_uid_0=test_uid_0.astype('int').tolist()
		test_uid_1=test_uid_1.astype('int').tolist()

		for name in clf_name:
			prob=[]
			real=[]
			prob_1=[]
			prob_0=[]

			#读取某分类器预测结果
			column_dict=self.load_clf_file(level,name)
			#排序
			column_dict2=sorted(column_dict.items(),key=lambda d:d[1])

			clf=[
				#'log_move_lr_sag',
				#'log_move_lr_newton',
				# 'log_move_lr_lbfgs',
				#  'log_move_lr_liblinear',
				#'log_move_rf100',
				# 'log_move_rf200',
				# 'log_move_rf500',
				# 'log_move_rf1000',
				# 'log_move_gbdt20',
				# 'log_move_gbdt50',
				#'log_move_gbdt100',
				# 'log_move_ada20',
				# 'log_move_ada50',
				'log_move_ada100',
				#'log_move_xgb2000',
				# 'log_move_xgb2500',
				#'log_move_xgb2000_2',
				# 'log_move_xgb2500_2'

			]
			ranks=[]
			for f_name in clf:
				rank=self.level_ranks('level_two',f_name)
				ranks.append(rank)

			column_ranks=self.level_ranks(level,name)

			i=0
			aa=0
			r_lr=0
			one_diff=[]
			zero_diff=[]
			one_index=[]
			zero_index=[]
			#选择区间进行blend

			# xgb_ranks_true=[]
			# xgb_ranks_false=[]
			# lr_ranks_true=[]
			# lr_ranks_false=[]
			# for k in range(21):
			# 	xgb_ranks_true.append(0)
			# 	xgb_ranks_false.append(0)
			# 	lr_ranks_true.append(0)
			# 	lr_ranks_false.append(0)
			# print xgb_ranks_true

			for uid, score in column_dict2:
				# if i<2000:
				# 	i+=1
				# 	continue
				diff=0
				for rank in ranks:
					diff+=column_ranks[uid][0]-rank[uid][0]
				#diff=diff/4
				if diff>500: #700+i*0.58
					column_dict[uid]=0
					r_lr+=1
					if uid in test_uid_1:
						print "no!!!"

				# rank_d=int(column_ranks[uid][0]/300)
				# rank_d2=int(rank[uid][0]/300)
				if uid in test_uid_0:
					zero_diff.append(diff)
					zero_index.append(i)
					aa+=1
					
					# xgb_ranks_true[rank_d]+=1
					# lr_ranks_true[rank_d2]+=1

					pass
				else:
					one_diff.append(diff)
					one_index.append(i)
					# xgb_ranks_false[rank_d]+=1
					# lr_ranks_false[rank_d2]+=1
					pass
					
				i+=1
				# if i>400:
				# 	break


			# print xgb_ranks_true
			# print xgb_ranks_false
			# print lr_ranks_true
			# print lr_ranks_false

			print aa
			print 'r_lr:',r_lr

			#计算blend后的AUC
			for uid,score in column_dict.items():
				prob.append(score)
				if uid in test_uid_0:
					real.append(0)
					prob_0.append(score)
				elif uid in test_uid_1:
					real.append(1)
					prob_1.append(score)
				else:
					print "error"

			auc_score=metrics.roc_auc_score(real,prob)
			print name,"  "," auc:",auc_score	
			print '0:',max(prob_0),min(prob_0)
			print "1:",max(prob_1),min(prob_1)

			#绘制不同分类器的排名差结果
			idex=0
			self.print_diff(zero_diff[idex:],zero_index[idex:],one_diff[idex:],one_index[idex:])
			return

	def print_diff(self,zero_diff,zero_index,one_diff,one_index):
		"""
		:type zero_diff: List[int] 0类的排名分布差
		:type zero_index: List[int] 0类的下标
		:type one_diff: List[int] 1类的排名分布差
		:type one_index: List[int] 1类的下标
		"""
		plt.scatter(zero_index,zero_diff)
		plt.scatter(one_index,one_diff,c='red')
		plt.show()

	def level_ranks(self,level,name):
		"""
		返回不同分类样本在本分类器中的排名
		"""
		load_data_instance=load_data.Load_data(self.config)
		X_0,test_X_0,X_1,test_X_1,uid_0,test_uid_0,uid_1,test_uid_1=load_data_instance.train_test_xy(1)

		test_uid_0=test_uid_0.astype('int').tolist()
		test_uid_1=test_uid_1.astype('int').tolist()

		ranks={}
		column_dict=self.load_clf_file(level,name)
		column_dict2=sorted(column_dict.items(),key=lambda d:d[1])
		i=0

		for uid, score in column_dict2:
			rank=ranks.get(uid,[])
			rank.append(i)
			ranks[uid]=rank
			i+=1

		return ranks

def main():
	config_instance=Config('log')
	config_instance.path_predict=config_instance.path+'predict_local/' #测试输出文件夹
	level='level_one'
	clf_name=[
		#'log_move_lr_sag',
		#'log_move_lr_newton',
		# 'log_move_lr_lbfgs',
		# 'log_move_lr_liblinear',
		# 'log_move_rf100',
		# 'log_move_rf200',
		# 'log_move_rf500',
		#'log_move_rf1000',
		# 'log_move_gbdt20',
		# 'log_move_gbdt50',
		 #'log_move_gbdt100',
		# 'log_move_ada20',
		# 'log_move_ada50',
		#'log_move_ada100',
		'log_move_xgb2000',
		#'log_move_xgb2500',
		# 'log_move_xgb2000_2',
		# 'log_move_xgb2500_2',
		#'log_move_ridge_part'

	]
	predict_data_instance=Local_predict_verify(config_instance,level,clf_name)
	predict_data_instance.level_data()
	pass

if __name__ == '__main__':
	main()