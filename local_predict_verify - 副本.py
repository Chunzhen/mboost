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
		#print test_uid_0
		#return

		for name in clf_name:
			prob=[]
			real=[]
			prob_1=[]
			prob_0=[]
			column_dict=self.load_clf_file(level,name)
			column_dict2=sorted(column_dict.items(),key=lambda d:d[1])

			lr_ranks=self.level_ranks('log_move_lr_sag')
			column_ranks=self.level_ranks(name)
			#print lr_ranks
			i=0

			print len(column_dict2)
			#return
			aa=0
			r_lr=0
			for uid, score in column_dict2:
				if i<2000:
					i+=1
					continue
				t_x,t_y=[],[]
				k=0
				loss_index=[]
				for j in range(len(test_X[test_uid_dict[str(12965)]])):
					if test_X[test_uid_dict[str(12965)]][j]!=-1 and test_X[test_uid_dict[str(uid)]][j]!=-1:
						t_x.append(test_X[test_uid_dict[str(12965)]][j])
						t_y.append(test_X[test_uid_dict[str(uid)]][j])

					if test_X[test_uid_dict[str(uid)]][j]==-1 or test_X[test_uid_dict[str(uid)]][j]==-2 or test_X[test_uid_dict[str(uid)]][j]==0:
						k+=1
						loss_index.append(j)

				# if k<200:
				# 	i+=1
				# 	continue
				cor=np.corrcoef(t_x,t_y)[0,1]
				if lr_ranks[uid][0]<500:
					#print 'bingo'

					column_dict[uid]=0
					# if uid==18893 or uid==19444:
					# 	column_dict[uid]=1
					r_lr+=1
					if uid in test_uid_1:
						print "no!!!"
				if uid in test_uid_0:
					print "0:",uid," ",score,' ',column_ranks[uid],' ',lr_ranks[uid],'k:',k	
					#print '0:','k:',k
					aa+=1			
					pass
				else:
					print "bingo 1:",uid," ",score,' ',column_ranks[uid],' ',lr_ranks[uid],',',np.corrcoef(t_x,t_y)[0,1],'k:',k	
					#print 'bingo 1, k:',k
					#print loss_index
					pass

				#print 'k:',k
				#print ""
				# if i<91:
				# 	# if  i==1:
				# 	# 	column_dict[uid]=1
				# 	i+=1
				# 	continue
				#column_dict[uid]=0
				
				
				# if uid==12935:
				# 	print name," ",i
				# 	break
				# if  i%2==1:
				# 	column_dict[uid]=1

				# if i>2990:
				# 	if uid in test_uid_0:
				# 		print 'change 0'
				# 		column_dict[uid]=0
				# 	else:
				# 		#print 'change 1'
				# 		pass
					
				i+=1
				if i>2100:
					break

			print aa
			print 'r_lr:',r_lr

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

			prob_0=sorted(prob_0)
			#print prob_0
			for i in range(len(prob_0)):
				if prob_0[i]>=0.04:
					print i 
					break

			return

	def level_ranks(self,name):
		level=self.level

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
	level='level_one'
	clf_name=[
		#'log_move_lr_sag',
		# 'log_move_lr_newton',
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
		# 'log_move_xgb2500_2'

	]
	predict_data_instance=Local_predict_verify(config_instance,level,clf_name)
	predict_data_instance.level_data()
	pass

if __name__ == '__main__':
	main()