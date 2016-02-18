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
			d[str(reader[0][i])]=reader[1][i]
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

		#X_00,test_X_00,X_11,test_X_11,uid_00,test_uid_00,uid_11,test_uid_11=load_data_instance.train_test_xy(1)
		#uids=np.hstack((uid_00,uid_11))
		#print len(uids)+len(test_uid_00)+len(test_uid_11)
		#y=np.hstack((np.ones(len(X_00)),np.zeros(len(X_11))))

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
		# print len(X_0)
		# print len(X_00)
		# return 
		# print uid_0[0]
		# print X_1[1]
		# print uid_1[1]
		#X_0=np.hstack((X_00,np.array(X_0)))
		#X_1=np.hstack((X_11,np.array(X_1)))
		return np.array(X_0),np.array(X_1),np.array(uid_0),np.array(uid_1)

	# def train_test_xy(self,random_state):
	# 	X_0,X_1,uid_0,uid_1=self.level_data()
	# 	train_X_0,test_X_0,train_uid_0,test_uid_0=train_test_split(X_0,uid_0,test_size=0.2,random_state=random_state)
	# 	train_X_1,test_X_1,train_uid_1,test_uid_1=train_test_split(X_1,uid_1,test_size=0.2,random_state=random_state)
	# 	return train_X_0,test_X_0,train_X_1,test_X_1,train_uid_0,test_uid_0,train_uid_1,test_uid_1

def  main():
	config_instance=Config('log_move')
	level='level_two'
	clf_name=[
		'log_move_lr_sag',
		'log_move_lr_newton',
		'log_move_lr_lbfgs',
		'log_move_lr_liblinear',
		'log_move_rf100',
		'log_move_rf200',
		'log_move_rf500',
		'log_move_rf1000',
		'log_move_gbdt20',
		'log_move_gbdt50',
		'log_move_gbdt100',
		'log_move_ada20',
		'log_move_ada50',
		'log_move_ada100',
		'log_move_xgb2000',
		'log_move_xgb2500',
		'log_move_xgb2000_2',
		'log_move_xgb2500_2'
	]
	load_data_instance=Load_train_data(config_instance,'level_two',clf_name)
	X_0,X_1,uid_0,uid_1=load_data_instance.level_data()
	pass

if __name__ == '__main__':
	main()
	

