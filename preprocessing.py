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
	def __init__(self,config):
		self.config=config
		pass

	def features_type(self):
		reader=pd.read_csv(self.config.path_feature_type,iterator=False,delimiter=',',encoding='utf-8')
		features=reader
		return features

	def scale_X(self):
		types=self.features_type()
		use=['uid']
		category_use=[]

		for i,t in enumerate(types['type']):
			if t=='category':
				category_use.append(types['feature'][i])
			else:
				use.append(types['feature'][i])

		train_reader=pd.read_csv(self.config.path_origin_train_x,iterator=False,delimiter=',',usecols=tuple(use),encoding='utf-8')
		test_reader=pd.read_csv(self.config.path_origin_predict_x,iterator=False,delimiter=',',usecols=tuple(use),encoding='utf-8')

		len_train=len(train_reader)
		len_predict=len(test_reader)

		reader=pd.concat([train_reader,test_reader])
		data=np.array(reader)

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

		uid=np.array(data[:,0],dtype='int')
		uid=uid.astype('str')	

		category_train_reader=pd.read_csv(self.config.path_origin_train_x,iterator=False,delimiter=',',usecols=tuple(category_use),encoding='utf-8')
		category_reader=pd.read_csv(self.config.path_origin_predict_x,iterator=False,delimiter=',',usecols=tuple(category_use),encoding='utf-8')

		len_train=len(category_train_reader)
		len_predict=len(category_reader)

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
		
		X=np.hstack((X,dummys))
		X_train=X[:len_train]
		X_predict=X[len_train:(len_train+len_predict)]

		
		pd.DataFrame(X_train).to_csv(self.config.path_train_x,sep=',',mode='wb',header=None,index=False)
		pd.DataFrame(X_predict).to_csv(self.config.path_predict_x,sep=',',mode='wb',header=None,index=False)
		pd.DataFrame(uid).to_csv(self.config.path_uid,sep=',',mode='wb',header=None,index=False)
		print self.config.scale_level1+"\n"

	def standard_scale(self,X):
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
	#scales=['log','log_move','standard','normalize','min_max','median','log_move_standard']
	scales=['log_standard']
	threads=[]
	for x in scales:
		config_instance=Config(x)
		preprocessing_instance=Preprocessing(config_instance)
		threads.append(threading.Thread(target=preprocessing_instance.scale_X))

	for t in threads:
		t.start()

	for t in threads:
		t.join()


	pass


def main():
	scale_wrapper()

if __name__ == '__main__':
	reload(sys)
	sys.setdefaultencoding('utf8')
	start=datetime.now()
	main()
	end=datetime.now()
	print "All Run time:"+str(float((end-start).seconds)/60.0)+"min / "+str(float((end-start).seconds))+"s"

