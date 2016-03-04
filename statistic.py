#! /usr/bin/env python
# -*- coding:utf-8 -*-
# 不同特征的数据统计

import sys
import os
import numpy as np
import pandas as pd
import math
import load_data
from config import Config

class Statistic(object):
	def __init__(self,config):
		self.config=config

	def features_type(self):
		reader=pd.read_csv(self.config.path_feature_type,iterator=False,delimiter=',',encoding='utf-8')
		features=reader
		return features

	def train_y(self):
		reader=pd.read_csv(self.config.path_train_y,iterator=False,delimiter=',',encoding='utf-8')
		data=np.array(reader)
		y=np.ravel(data[:,1:])
		uid=np.array(data[:,0],dtype='str')
		return y,uid

	def log_scale_move(self,X):
		n,m=X.shape
		for i in range(m):
			column=X[:,i]

			c_max=np.max(column)
			c_min=np.min(column)
			for j in range(n):
				column[j]=math.log10(column[j]-c_min+1)**2

			X[:,i]=column
		return X

	def train_origin_X(self):
		types=self.features_type()
		use=['uid']
		category_use=[]
		j=0
		for i,t in enumerate(types['type']):
			
			if types['type'][i]=='numeric':
				j+=1
				use.append(types['feature'][i])

		print j
		#return

		reader=pd.read_csv(self.config.path_origin_train_x,iterator=False,delimiter=',',usecols=tuple(use),encoding='utf-8')
		#print reader
		data=np.array(reader)
		X=data[:,1:]
		#X=self.log_scale_move(X)
		uid=np.array(data[:,0],dtype='int')
		uid=uid.astype('str')
		y,uid=self.train_y()

		#f=open(self.config.path+'statistic/cor.csv','wb')
		j=0
		k=0
		for i in range(len(X)):
			k=0
			for j in range(len(X[i])):
				if X[i,j]==-1 or X[i,j]==-2:
					k+=1

			if k>200:
				print y[i],' ','k:',k





		#print 'big one:',j

		#f.close()

		# for i in range(len(X[0])):
		# 	sorted_col=self.column(X[:,i])
		# 	if sorted_col[0][0]-sorted_col[1][0]>sorted_col[0][0]*0.7 and sorted_col[0][1]<10 and sorted_col[1][1]<10:
		# 		print use[i]
		# 		#print sorted_col[:10]
		# 		for j in range(len(X[:,i])):
		# 			if abs(sorted_col[0][0]-X[j,i])<2:
		# 				print uid[i],' ',y[i]
		# 				break

		# print X.shape
		#return X,uid

	def column(self,col):
		col=col.astype('int')
		d={}
		for v in col:
			n=d.get(v,0)
			d[v]=n+1

		sorted_col=sorted(d.items(),key=lambda d:d[0],reverse=True)
		return sorted_col

def  main():
	config_instance=Config('log_move')
	instance=Statistic(config_instance)
	instance.train_origin_X()
	pass

if __name__ == '__main__':
	main()