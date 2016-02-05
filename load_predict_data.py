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

class Load_predict_data(object):
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
		predict_X,uids=load_data_instance.predict_X()

		d={}
		for name in clf_name:
			column_dict=self.load_clf_file(level,name)
			for uid in uids:
				temp=d.get(uid,[])
				temp.append(column_dict[uid])
				d[uid]=temp
		
		X=[]

		for i in range(len(uids)):
			X.append(d[uids[i]])

		print X[0]
		print uids[0]
		print X[1]
		print uids[1]
		return np.array(X),np.array(uids)

def main():
	config_instance=Config('log_move')
	level='level_two'
	clf_name=[
		# 'log_move_lr_sag',
		# 'log_move_lr_newton',
		# 'log_move_lr_lbfgs',
		# 'log_move_lr_liblinear',
		# 'log_move_rf100',
		# 'log_move_rf200',
		# 'log_move_rf500',
		# 'log_move_rf1000',
		# 'log_move_gbdt20',
		# 'log_move_gbdt50',
		# 'log_move_gbdt100',
		# 'log_move_ada20',
		# 'log_move_ada50',
		# 'log_move_ada100',
		# 'log_move_xgb2000',
		# 'log_move_xgb2500',
		'log_move_xgb2000_2',
		'log_move_xgb2500_2'
	]
	predict_data_instance=Load_predict_data(config_instance,'level_one',clf_name)
	predict_X,predict_uid=predict_data_instance.level_data()
	pass

if __name__ == '__main__':
	main()