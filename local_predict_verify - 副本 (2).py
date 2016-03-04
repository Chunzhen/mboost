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
		load_data_instance=load_data.Load_data(self.config)
		X_0,test_X_0,X_1,test_X_1,uid_0,test_uid_0,uid_1,test_uid_1=load_data_instance.train_test_xy(1)
		predict_X,uids=load_data_instance.predict_X()
		test_X=np.vstack((test_X_0,test_X_1))
		test_uid=(np.hstack((test_uid_0,test_uid_1))).tolist()
		test_uid_dict={}
		for i in range(len(test_uid)):
			test_uid_dict[test_uid[i]]=i

		d={}
		test_uid_0=test_uid_0.astype('int').tolist()
		test_uid_1=test_uid_1.astype('int').tolist()

		for name in clf_name:
			prob=[]
			real=[]
			prob_1=[]
			prob_0=[]
			column_dict=self.load_clf_file(level,name)
			column_dict2=sorted(column_dict.items(),key=lambda d:d[1])

			clf=[
				'log_move_lr_sag',
				#'log_move_lr_newton',
				# 'log_move_lr_lbfgs',
				 #'log_move_lr_liblinear',
				# 'log_move_rf100',
				# 'log_move_rf200',
				# 'log_move_rf500',
				#'log_move_rf1000',
				# 'log_move_gbdt20',
				# 'log_move_gbdt50',
				#'log_move_gbdt100',
				#'log_move_ada20',
				# 'log_move_ada50',
				#'log_move_ada100',
				#'log_move_xgb2000',
				#'log_move_xgb2500',
				#'log_move_xgb2000_2',
				#'log_move_xgb2500_2',
				#'log_move_ridge_part'

			]
			ranks=[]
			for f_name in clf:
				rank=self.level_ranks('level_two',f_name)
				ranks.append(rank)

			# dicts=[]
			# for f_name in clf:
			# 	d=self.load_clf_file(level,f_name)
			# 	dicts.append(d)

			column_ranks=self.level_ranks(level,name)
			#print lr_ranks
			i=0

			print len(column_dict2)
			#return
			aa=0
			r_lr=0
			one_diff=[]
			zero_diff=[]
			one_index=[]
			zero_index=[]
			for uid, score in column_dict2:
				if i<0:
					i+=1
					continue
				diff=0
				# for d in dicts:
				# 	column_dict[uid]+=d[uid]

				#column_dict[uid]=column_dict[uid]/4.0
				for rank in ranks:
					diff+=column_ranks[uid][0]-rank[uid][0]
				diff=diff/4
				if diff>-100:
					column_dict[uid]=0
					r_lr+=1
					if uid in test_uid_1:
						print "no!!!"

				# if ((column_ranks[uid][0]-lr_ranks[uid][0])+(column_ranks[uid][0]-xgb_ranks[uid][0]))>1700:
				# 	column_dict[uid]=0

				# 	r_lr+=1
				# 	if uid in test_uid_1:
				# 		print "no!!!"
				if uid in test_uid_0:
					#print "0:",uid," ",score,' ',column_ranks[uid],' ',lr_ranks[uid],' ',column_ranks[uid][0]-lr_ranks[uid][0]
					zero_diff.append(diff)
					zero_index.append(i)
					aa+=1			
					pass
				else:
					#print "bingo 1:",uid," ",score,' ',column_ranks[uid],' ',lr_ranks[uid],' ',column_ranks[uid][0]-lr_ranks[uid][0]
					one_diff.append(diff)
					one_index.append(i)
					pass
					
				i+=1
				# if i>400:
				# 	break

			print aa
			print 'r_lr:',r_lr
			#print 'one:',max(one_diff),' ',min(one_diff)
			#print 'zero:',max(zero_diff),' ',min(zero_diff)

			
			#return

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

			idex=0
			self.print_diff(zero_diff[idex:],zero_index[idex:],one_diff[idex:],one_index[idex:])
	def print_diff(self,zero_diff,zero_index,one_diff,one_index):
		plt.scatter(zero_index,zero_diff)
		plt.scatter(one_index,one_diff,c='red')
		plt.show()
		pass
	def level_ranks(self,level,name):
		#level=self.level
		#level='level_three'

		load_data_instance=load_data.Load_data(self.config)
		X_0,test_X_0,X_1,test_X_1,uid_0,test_uid_0,uid_1,test_uid_1=load_data_instance.train_test_xy(1)
		predict_X,uids=load_data_instance.predict_X()

		d={}
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


	def output_rank(self,ranks,path):
		f=open(path,'wb')
		for uid,rank in ranks.items():
			f.write(str(uid))
			for r in rank:
				f.write(','+str(r))
			f.write('\n')
		f.close()

def main():
	config_instance=Config('log')
	level='level_two'
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