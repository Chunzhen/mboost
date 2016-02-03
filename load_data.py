#! /usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import time

import config

class Load_data(object):
	def __init__(self):
		self.__scale=config.scale_level1

	def features_type(self):
		reader=pd.read_csv(config.path_feature_type,iterator=False,delimiter=',',encoding='utf-8')
		features=reader
		return features

	def train_X(self):
		X=pd.read_csv(config.path_train_x,iterator=False,delimiter=',',encoding='utf-8',header=None)
		uid=pd.read_csv(config.path_uid,iterator=False,delimiter=',',encoding='utf-8',header=None)
		X=np.array(X,dtype="float32")
		X=np.nan_to_num(X)
		uid=np.array(uid)
		print X.shape
		return X,uid

	def train_y(self):
		reader=pd.read_csv(config.path_train_y,iterator=False,delimiter=',',encoding='utf-8')
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


	def predict_X(self):
		X=pd.read_csv(config.path_predict_x,iterator=False,delimiter=',',encoding='utf-8',header=None)
		uid=pd.read_csv(config.path_uid,iterator=False,delimiter=',',encoding='utf-8',header=None)
		X=np.array(X,dtype="float32")
		X=np.nan_to_num(X)
		uid=np.array(uid).ravel()
		#print X.shape
		train_reader=pd.read_csv('data/train_x.csv',iterator=False,delimiter=',',usecols=tuple(['uid']),encoding='utf-8')
		test_reader=pd.read_csv('data/test_x.csv',iterator=False,delimiter=',',usecols=tuple(['uid']),encoding='utf-8')
		len_train=len(train_reader)
		len_predict=len(test_reader)
		return X,uid[len_train:(len_train+len_predict)]


def main():

	pass

if __name__ == '__main__':
	main()