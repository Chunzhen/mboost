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
	"""
	:class Load_data
	:读取第一层数据类，将Preprocessing类输出的数据读取
	"""
	def __init__(self,config):
		"""
		:type config: Config
		:初始化配置信息
		"""
		self.config=config

	def features_type(self):
		"""
		加载特征列的类型:numeric or category
		"""
		reader=pd.read_csv(self.config.path_feature_type,iterator=False,delimiter=',',encoding='utf-8')
		features=reader
		return features

	def train_X(self):
		"""
		读取self.config.path_train_x下的特征矩阵数据
		"""
		X=pd.read_csv(self.config.path_train_x,iterator=False,delimiter=',',encoding='utf-8',header=None)
		uid=pd.read_csv(self.config.path_uid,iterator=False,delimiter=',',encoding='utf-8',header=None)
		X=np.array(X,dtype="float32")
		X=np.nan_to_num(X)
		uid=np.array(uid)
		print X.shape
		return X,uid

	def train_y(self):
		"""
		读取类标签列数据
		"""
		reader=pd.read_csv(self.config.path_train_y,iterator=False,delimiter=',',encoding='utf-8')
		data=np.array(reader)
		y=np.ravel(data[:,1:])
		uid=np.array(data[:,0],dtype='str')
		return y,uid

	def train_xy(self):
		"""
		将类0与类1拆分到不同数组
		特别地，由于XGBoost分类器对类标签反转的情况训练更佳，所以负类变为1，正类变为0
		在预测线上结果时，需要对所有输出结果进行1-predict处理
		"""
		X,uid=self.train_X()
		y,uid=self.train_y()

		X_0=[]
		X_1=[]
		uid_0=[]
		uid_1=[]
		for i in range(len(y)):
			if y[i]==0: #反转类标签
				X_1.append(X[i])
				uid_1.append(uid[i])
			else:
				X_0.append(X[i])
				uid_0.append(uid[i])
		return np.array(X_0),np.array(X_1),np.array(uid_0),np.array(uid_1)

	def train_test_xy(self,random_state):
		"""
		将原来的训练数据分离出20%作为本地验证集，对多层的训练结果进行独立的验证
		"""
		X_0,X_1,uid_0,uid_1=self.train_xy() #读取全部训练数据
		train_X_0,test_X_0,train_uid_0,test_uid_0=train_test_split(X_0,uid_0,test_size=0.2,random_state=random_state)
		train_X_1,test_X_1,train_uid_1,test_uid_1=train_test_split(X_1,uid_1,test_size=0.2,random_state=random_state)
		return train_X_0,test_X_0,train_X_1,test_X_1,train_uid_0,test_uid_0,train_uid_1,test_uid_1

	def predict_X(self):
		"""
		读取预测集的数据
		"""
		X=pd.read_csv(self.config.path_predict_x,iterator=False,delimiter=',',encoding='utf-8',header=None)
		X=np.array(X)

		test_reader=pd.read_csv(self.config.path_origin_predict_x,iterator=False,delimiter=',',usecols=tuple(['uid']),encoding='utf-8')
		uid=np.array(test_reader).ravel()
		return X,uid


def main():
	"""
	本地测试函数
	"""
	config_instance=config.Config('log_move')
	load_data_instance=Load_data(config_instance)
	predict_X,uid=load_data_instance.predict_X()
	train_X_0,test_X_0,train_X_1,test_X_1,train_uid_0,test_uid_0,train_uid_1,test_uid_1=load_data_instance.train_test_xy(1)
	print len(train_uid_0)+len(test_uid_0)+len(train_uid_1)+len(test_uid_1)
	print len(X_0),' ',len(X_1),' ',len(uid_0),' ',len(uid_1)
	pass

if __name__ == '__main__':
	main()