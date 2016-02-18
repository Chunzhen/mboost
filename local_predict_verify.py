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

		d={}
		test_uid_0=test_uid_0.astype('int').tolist()
		test_uid_1=test_uid_1.astype('int').tolist()
		#print test_uid_0
		#return

		for name in clf_name:
			prob=[]
			real=[]
			column_dict=self.load_clf_file(level,name)
			for uid,score in column_dict.items():
				prob.append(score)
				if uid in test_uid_0:
					real.append(0)
				elif uid in test_uid_1:
					real.append(1)
				else:
					print "error"

			auc_score=metrics.roc_auc_score(real,prob)
			print name,"  "," auc:",auc_score	


def main():
	config_instance=Config('log_move')
	level='level_one'
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
	predict_data_instance=Local_predict_verify(config_instance,level,clf_name)
	predict_data_instance.level_data()
	pass

if __name__ == '__main__':
	main()