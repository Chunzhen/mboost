#! /usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import time
import config

from sklearn.cross_validation import train_test_split

class Load_data(object):
	def __init__(self,config):
		self.config=config

	def features_type(self):
		reader=pd.read_csv(self.config.path_feature_type,iterator=False,delimiter=',',encoding='utf-8')
		features=reader
		return features

	def train_X(self):
		X=pd.read_csv(self.config.path_train_x,iterator=False,delimiter=',',encoding='utf-8',header=None)
		uid=pd.read_csv(self.config.path_uid,iterator=False,delimiter=',',encoding='utf-8',header=None)
		X=np.array(X,dtype="float32")
		X=np.nan_to_num(X)
		uid=np.array(uid)
		print X.shape
		return X,uid

	def train_y(self):
		reader=pd.read_csv(self.config.path_train_y,iterator=False,delimiter=',',encoding='utf-8')
		data=np.array(reader)
		y=np.ravel(data[:,1:])
		uid=np.array(data[:,0],dtype='str')
		return y,uid

	def train_xy(self):
		X,uid=self.train_X()
		y,uid=self.train_y()

		#print X.shape
		#print y.shape
		X_0=[]
		X_1=[]
		uid_0=[]
		uid_1=[]
		for i in range(len(y)):
			if y[i]==0:
				X_1.append(X[i])
				uid_1.append(uid[i])
			else:
				X_0.append(X[i])
				uid_0.append(uid[i])
		return np.array(X_0),np.array(X_1),np.array(uid_0),np.array(uid_1)

	def train_test_xy(self,random_state):
		X_0,X_1,uid_0,uid_1=self.train_xy()
		train_X_0,test_X_0,train_uid_0,test_uid_0=train_test_split(X_0,uid_0,test_size=0.2,random_state=random_state)
		train_X_1,test_X_1,train_uid_1,test_uid_1=train_test_split(X_1,uid_1,test_size=0.2,random_state=random_state)
		return train_X_0,test_X_0,train_X_1,test_X_1,train_uid_0,test_uid_0,train_uid_1,test_uid_1

	def part_uid_xy(self,level,name):
		reader=pd.read_csv(self.config.path_train+level+'/'+name+'_part_uid.csv',iterator=False,delimiter=',',header=None,encoding='utf-8')
		uidSet=set()
		for uid in reader[0]:
			uidSet.add(str(uid))

		X,uid=self.train_X()
		y,uid=self.train_y()

		#print X.shape
		#print y.shape
		X_0=[]
		X_1=[]
		uid_0=[]
		uid_1=[]
		for i in range(len(y)):
			if uid[i] in uidSet:
				if y[i]==0:
					X_1.append(X[i])
					uid_1.append(uid[i])
				else:
					X_0.append(X[i])
					uid_0.append(uid[i])
		return np.array(X_0),np.array(X_1),np.array(uid_0),np.array(uid_1)

	def predict_X(self):
		X=pd.read_csv(self.config.path_predict_x,iterator=False,delimiter=',',encoding='utf-8',header=None)
		X=np.array(X)

		test_reader=pd.read_csv(self.config.path_origin_predict_x,iterator=False,delimiter=',',usecols=tuple(['uid']),encoding='utf-8')
		uid=np.array(test_reader).ravel()
		return X,uid


def main():
	config_instance=config.Config('log_move')
	load_data_instance=Load_data(config_instance)
	#predict_X,uid=load_data_instance.predict_X()
	#train_X_0,test_X_0,train_X_1,test_X_1,train_uid_0,test_uid_0,train_uid_1,test_uid_1=load_data_instance.train_test_xy(1)
	#print len(train_uid_0)+len(test_uid_0)+len(train_uid_1)+len(test_uid_1)
	X_0,X_1,uid_0,uid_1=load_data_instance.part_uid_xy('level_one','log_move_xgb1000_test_i')
	print len(X_0),' ',len(X_1),' ',len(uid_0),' ',len(uid_1)
	pass

if __name__ == '__main__':
	main()