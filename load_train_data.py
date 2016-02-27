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

from sklearn.cross_validation import train_test_split

class Load_train_data(object):
	def __init__(self,config,level,clf_name):
		self.config=config
		self.level=level
		self.__clf_name=clf_name

	def load_clf_file(self,level,name):
		reader=pd.read_csv(self.config.path_train+level+'/'+name+'.csv',iterator=False,delimiter=',',encoding='utf-8',header=None)
		d={}
		for i in range(len(reader[0])):
			d[str(reader[0][i])]=np.log10(reader[1][i])
			#d[str(reader[0][i])]=reader[1][i]
		return d

	def load_clf_score(self,level,name):
		reader=pd.read_csv(self.config.path_train+level+'/'+name+'_score.csv',iterator=False,delimiter=',',encoding='utf-8',header=None)
		return np.mean(reader[0])

	def level_data(self):
		level=self.level
		clf_name=self.__clf_name
		load_data_instance=load_data.Load_data(self.config)
		y,uids=load_data_instance.train_y()
		X_00,X_11,uid_00,uid_11=load_data_instance.train_xy()

		# X_00,test_X_00,X_11,test_X_11,uid_00,test_uid_00,uid_11,test_uid_11=load_data_instance.train_test_xy(1)
		# uids=np.hstack((uid_00,uid_11))
		# print len(uids)+len(test_uid_00)+len(test_uid_11)
		# y=np.hstack((np.ones(len(X_00)),np.zeros(len(X_11))))

		column_important=[]
		d={}
		for name in clf_name:
			column_dict=self.load_clf_file(level,name)
			#print len(column_dict)
			column_score=self.load_clf_score(level,name)
			column_important.append(column_score)
			print name,"  ",column_score

			for uid in uids:
				temp=d.get(uid,[])
				temp.append(column_dict[uid])
				d[uid]=temp
		
		# ranks=self.level_ranks()
		# for uid in uids:
		# 	d[uid].extend(ranks[uid])

		X_0=[]
		X_1=[]
		uid_0=[]
		uid_1=[]

		for i in range(len(y)):
			if y[i]==0:
				X_1.append(d[uids[i]])
				uid_1.append(uids[i])
			else:
				X_0.append(d[uids[i]])
				uid_0.append(uids[i])


		#print column_important
		print np.array(X_0).shape
		# print len(X_00)
		# return 
		# print uid_0[0]
		# print X_1[1]
		# print uid_1[1]
		#X_0=np.hstack((X_00,np.array(X_0)))
		#X_1=np.hstack((X_11,np.array(X_1)))
		return np.array(X_0),np.array(X_1),np.array(uid_0),np.array(uid_1)

	def level_ranks(self):
		level=self.level
		ftype='log_move'
		clf_name=[
			ftype+'_lr_sag',
			ftype+'_lr_newton',
			ftype+'_lr_lbfgs',
			ftype+'_lr_liblinear',
			ftype+'_rf100',
			ftype+'_rf200',
			ftype+'_rf500',
			ftype+'_rf1000',
			ftype+'_gbdt20',
			ftype+'_gbdt50',
			ftype+'_gbdt100',
			ftype+'_ada20',
			ftype+'_ada50',
			ftype+'_ada100',
		]

		ranks={}
		column_dict_out=self.load_clf_file(level,ftype+'_xgb2500_2')
		column_dict2_out=sorted(column_dict_out.items(),key=lambda d:d[1])
		rank_out={}
		i=1
		for uid,score in column_dict2_out:
			rank_out[uid]=i
			i+=1

		for name in clf_name:
			column_dict=self.load_clf_file(level,name)
			column_dict2=sorted(column_dict.items(),key=lambda d:d[1])
			i=0

			for uid, score in column_dict2:
				rank=ranks.get(uid,[])
				rank.append(float(i)/rank_out[uid])
				ranks[uid]=rank
				i+=1

		return ranks
		#self.output_rank(ranks,self.config.path_train+level+'/'+'ranks'+'.csv')

	def output_rank(self,ranks,path):
		f=open(path,'wb')
		for uid,rank in ranks.items():
			f.write(str(uid))
			for r in rank:
				f.write(','+str(r))
			f.write('\n')
		f.close()


def  main():
	ftype='log_move'
	config_instance=Config(ftype)
	level='level_one'
	# clf_name=[
	# 	ftype+'_lr_sag',
	# 	ftype+'_lr_newton',
	# 	ftype+'_lr_lbfgs',
	# 	ftype+'_lr_liblinear',
	# 	ftype+'_rf100',
	# 	ftype+'_rf200',
	# 	ftype+'_rf500',
	# 	ftype+'_rf1000',
	# 	ftype+'_gbdt20',
	# 	ftype+'_gbdt50',
	# 	ftype+'_gbdt100',
	# 	ftype+'_ada20',
	# 	ftype+'_ada50',
	# 	ftype+'_ada100',
	# 	ftype+'_xgb2000',
	# 	ftype+'_xgb2500',
	# 	ftype+'_xgb2000_2',
	# 	ftype+'_xgb2500_2'
	# ]
	clf_name=[]
	for i in range(42):
		clf_name.append(ftype+'_xgb1000_test_'+str(i))

	load_data_instance=Load_train_data(config_instance,level,clf_name)
	X_0,X_1,uid_0,uid_1=load_data_instance.level_data()
	#load_data_instance.level_ranks()
	pass

if __name__ == '__main__':
	main()
	

