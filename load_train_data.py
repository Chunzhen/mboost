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
		load_data_instance=load_data.Load_data()
		y,uids=load_data_instance.train_y()

		column_important=[]
		d={}
		for name in clf_name:
			column_dict=self.load_clf_file(level,name)
			column_score=self.load_clf_score(level,name)
			column_important.append(column_score)
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


		print column_important
		return np.array(X_0),np.array(X_1),np.array(uid_0),np.array(uid_1)
	

