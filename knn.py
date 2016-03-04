#! /usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import time

import load_data
import load_train_data
import load_predict_data
import copy
from mboost import Mboost

from config import Config

from numpy import tile
import operator
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
class Knn(object):
	"""
	:class Knn 
	knn分类器，分为正负类的参考标准不是k neighbors里面正负类的个数大的作为类标签
	而是只要k neighbors里面超过b个负类，即置为负类
	"""
	def __init__(self,config):
		"""
		:type config: Config
		:初始化配置信息
		"""
		self.config=config

	def knn_level(self,level,name,X_0,X_1,uid_0,uid_1):
		"""
		:type level: str 训练第几层
		:type name: str 分类器命名
		:type X_0: numpy.array 类别0特征矩阵
		:type X_1: numpy.array 类型1特征矩阵
		:type uid_0: List 类别0 uid
		:type uid_1: List 类别1 uid
		n folds训练
		"""
		boost=Mboost(self.config)
		n_folds=self.config.n_folds
		f0,f1=boost.fold(len(X_0),len(X_1),n_folds)

		predicts=[]
		test_uids=[]
		scores=[]

		for i in range(n_folds):
			train_index_0,test_index_0=f0[i][0],f0[i][1]
			train_index_1,test_index_1=f1[i][0],f1[i][1]

			train_1=X_1[train_index_1]
			test_1=X_1[test_index_1]

			train_0=X_0[train_index_0]
			test_0=X_0[test_index_0]

			test_uid_1=uid_1[test_index_1]
			test_uid_0=uid_0[test_index_0]

			y_train=np.hstack((np.ones(len(train_1)),np.zeros(len(train_0))))
			y_test=np.hstack((np.ones(len(test_1)),np.zeros(len(test_0))))

			test_uid=np.hstack((test_uid_1,test_uid_0))

			x_train=np.vstack((train_1,train_0))
			x_test=np.vstack((test_1,test_0))

			self.knn_classifier(x_train, y_train, x_test, y_test)

	def knn_classifier(self,x_train,y_train,x_test,y_test):
		"""
		:type x_train: numpy.array 训练特征矩阵
		:type y_train: List[int] 训练类标签
		:type x_test: numpy.array 测试特征矩阵
		:type y_test: List[int] 测试类标签
		单个模型训练预测
		"""
		y_prob=[]
		
		for i in range(len(x_test)):
			# if i <20:
			# 	continue
			# elif i>30:
			# 	break
			t=x_test[i]
			t_y=y_test[i]
			classCount=self.knn_one(x_train, y_train, t,30)
			#print 'y:',t_y,'...'
			#print classCount
			y_prob.append(float(classCount.get('1',0))/6.0)

		auc_score=metrics.roc_auc_score(y_test,y_prob)
		print auc_score

	def knn_one(self,x_train,y_train,t,k):
		"""
		:type x_train: numpy.array 训练特征矩阵
		:type y_train: List[int] 训练类标签
		:type t: numpy.array 测试特征矩阵
		:type k: k值
		选择k neighbors，并返回k neighbors里含有的负类neighbors的个数
		"""
		train_size=x_train.shape[0]
		diffMat=tile(t,(train_size,1))-x_train
		sqDiffMat=diffMat**2

		sqDistances=sqDiffMat.sum(axis=1)
		distances=sqDistances**0.5
		sortedDistIndicies=distances.argsort()
		classCount={}
		for i in range(k):
			voteIlabel=y_train[sortedDistIndicies[i]]
			classCount[str(int(voteIlabel))]=classCount.get(str(int(voteIlabel)),0)+1
		return classCount


def main():
	"""
	本地测试函数
	"""
	ftype='log_move'
	config_instance=Config(ftype)
	load_data_instance=load_data.Load_data(config_instance)
	#X_0,X_1,uid_0,uid_1=load_data_instance.train_xy()
	#本地训练，取出一份作本地验证
	X_0,test_X_0,X_1,test_X_1,uid_0,test_uid_0,uid_1,test_uid_1=load_data_instance.train_test_xy(1)
	level='level_one'

	classifier_instance=Knn(config_instance)
	classifier_instance.knn_level(level,'knn',X_0,X_1,uid_0,uid_1)
	pass

if __name__ == '__main__':
	main()


