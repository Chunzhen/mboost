#! /usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import time

from config import Config

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
import math
import threading
class Preprocessing(object):
	"""
	:class Preprocessing
	:数据预处理类，在原始数据集的基础上对数据集进行以下几种变换
	:min_max: 归一化
	:standard: 标准化
	:normalize: 规范化
	:median: 缺失值填充中位数
	:log: log10变换，如果为负数，则直接忽略
	:log_move: 将列特征整体向右移动本列最小值位，然后log10变换
	:log_move_cor: 过滤与类标签相关系数过低的列，然后作log10
	:log_move_standard: log_move与standard组合
	:log_standard: log与standard组合
	:独立出此类的好处是可以多线程处理特征，每次训练时只加载处理后的特征，节省训练时间
	"""
	def __init__(self,config):
		"""
		:type config: Config
		:初始化配置信息
		"""
		self.config=config
		pass

	def features_type(self):
		"""
		加载特征列的类型:numeric or category
		"""
		reader=pd.read_csv(self.config.path_feature_type,iterator=False,delimiter=',',encoding='utf-8')
		features=reader
		return features

	def load_cor_feature(self):
		"""
		加载相关系数绝对值超过0.01的特征列
		"""
		reader=pd.read_csv(self.config.path_cor,iterator=False,delimiter=',',encoding='utf-8',header=None)
		cor_features=set([])
		for i in range(len(reader[0])):
			if abs(reader[1][i])>=0.01:
				cor_features.add(reader[0][i])

		print 'cor_features:',len(cor_features)
		return cor_features

	def scale_X(self):
		"""
		特征变换，根据配置的self.config.scale_level1的值来确定变换类型
		"""
		types=self.features_type()
		use=['uid'] #存储numeric类型的特征
		category_use=[] #存储category类型的特性
		if self.config.scale_level1=='log_move_cor':
			cor_features=self.load_cor_feature()

		for i,t in enumerate(types['type']):
			if t=='category':
				category_use.append(types['feature'][i])
			else:
				if self.config.scale_level1=='log_move_cor':
					if types['feature'][i] in cor_features:
						use.append(types['feature'][i])
				else:
					use.append(types['feature'][i])

		#分别对训练数据与预测数据读取，统一处理
		train_reader=pd.read_csv(self.config.path_origin_train_x,iterator=False,delimiter=',',usecols=tuple(use),encoding='utf-8')
		test_reader=pd.read_csv(self.config.path_origin_predict_x,iterator=False,delimiter=',',usecols=tuple(use),encoding='utf-8')

		len_train=len(train_reader)
		len_predict=len(test_reader)

		reader=pd.concat([train_reader,test_reader])
		data=np.array(reader)

		#对numeric特征列进行变换
		X=data[:,1:]
		if self.config.scale_level1=='log':
			X=self.log_scale(X)
		elif self.config.scale_level1=='log_move':
			X=self.log_scale_move(X)
		elif self.config.scale_level1=='standard':	
			X=self.standard_scale(X)
		elif self.config.scale_level1=='normalize':
			X=self.normalizer_scale(X)
		elif self.config.scale_level1=='min_max':
			X=self.min_max_scale(X)
		elif self.config.scale_level1=='median':
			X=self.fill_scale(X,self.median_feature(X))
		elif self.config.scale_level1=='log_move_standard':
			X=self.log_scale_move(X)
			X=self.standard_scale(X)
		elif self.config.scale_level1=='log_standard':
			X=self.log_scale(X)
			X=self.standard_scale(X)
		elif self.config.scale_level1=='log_move_cor':
			X=self.log_scale_move(X)

		uid=np.array(data[:,0],dtype='int')
		uid=uid.astype('str')	

		#读取category的特征列
		category_train_reader=pd.read_csv(self.config.path_origin_train_x,iterator=False,delimiter=',',usecols=tuple(category_use),encoding='utf-8')
		category_reader=pd.read_csv(self.config.path_origin_predict_x,iterator=False,delimiter=',',usecols=tuple(category_use),encoding='utf-8')

		len_train=len(category_train_reader)
		len_predict=len(category_reader)

		#对category的特征列数据进行哑变量变换
		category_reader=pd.concat([category_train_reader,category_reader])
		dummys=pd.DataFrame()
		j=1
		for i in range(len(category_use)):
			temp_dummys=pd.get_dummies(category_reader[category_use[i]])
			if j==1:
				j+=1
				dummys=temp_dummys
			else:
				dummys=np.hstack((dummys,temp_dummys))
		
		#合并numeric和category的结果
		X=np.hstack((X,dummys))
		X_train=X[:len_train]
		X_predict=X[len_train:(len_train+len_predict)]

		#输出结果
		pd.DataFrame(X_train).to_csv(self.config.path_train_x,sep=',',mode='wb',header=None,index=False)
		pd.DataFrame(X_predict).to_csv(self.config.path_predict_x,sep=',',mode='wb',header=None,index=False)
		pd.DataFrame(uid).to_csv(self.config.path_uid,sep=',',mode='wb',header=None,index=False)
		print self.config.scale_level1+"\n"

	def standard_scale(self,X):
		"""
		:type X: numpy.array 特征矩阵
		:rtype X: numpy.array 变换后特征
		:特征变换工具函数
		"""
		scaler=StandardScaler()
		return scaler.fit_transform(X)

	def min_max_scale(self,X):
		scaler=MinMaxScaler()
		return scaler.fit_transform(X)

	def normalizer_scale(self,X):
		scaler=Normalizer()
		return scaler.fit_transform(X)

	#每个feature的中位数
	def median_feature(self,X):
		m,n=X.shape
		X_median=[]
		for i in range(n):
			median=np.median(X[:,i])
			X_median.append(median)
		return X_median

	#median 填充-1值
	def fill_scale(self,X,X_median):
		m,n=X.shape
		for i in range(m):
			for j in range(n):
				if X[i][j]==-1 or X[i][j]==-2:
					X[i][j]=X_median[j]
		return X

	def log_scale(self,X):
		m,n=X.shape
		for i in range(m):
			for j in range(n):
				if X[i][j]>0:
					X[i][j]=math.log10(X[i][j])
		return X

	def log_scale_move(self,X):
		n,m=X.shape
		for i in range(m):
			column=X[:,i]

			c_max=np.max(column)
			c_min=np.min(column)
			for j in range(n):
				column[j]=math.log10(column[j]-c_min+1)

			X[:,i]=column
		return X

def scale_wrapper():
	"""
	配置每个变换的值，加入线程集合，并发处理
	"""
	scales=['log','log_move','standard','normalize','min_max','median','log_move_standard']
	threads=[]
	for x in scales:
		config_instance=Config(x)
		preprocessing_instance=Preprocessing(config_instance)
		threads.append(threading.Thread(target=preprocessing_instance.scale_X))

	for t in threads:
		t.start()

	for t in threads:
		t.join()

def main():
	"""
	本地测试函数
	"""
	scale_wrapper()

if __name__ == '__main__':
	reload(sys)
	sys.setdefaultencoding('utf8')
	start=datetime.now()
	main()
	end=datetime.now()
	print "All Run time:"+str(float((end-start).seconds)/60.0)+"min / "+str(float((end-start).seconds))+"s"

