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

from config import Config

from sklearn.cross_validation import KFold
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import xgboost as xgb
class Mboost(object):
	"""
	:class Mboost
	:独立出训练与测试的类，多层模型时可调用相同类，达到代码重用
	"""
	def __init__(self,config):
		"""
		:type config: Config
		:初始化配置信息
		"""
		self.config=config
		pass

	def fold(self,len_0,len_1,n_folds):
		"""
		:type len_0: int
		:type len_1: int
		:type n_folds: int
		:rtype f0: List[List[int]]
		:rtype f1: List[List[int]]
		:将正类与负类分别分为n折，返回这n折每一折的下标集合
		"""
		random_state=self.config.fold_random_state #设置随机值
		kf0=KFold(n=len_0, n_folds=n_folds, shuffle=True,random_state=random_state)
		kf1=KFold(n=len_1,n_folds=n_folds, shuffle=True,random_state=random_state)
		f0=[]
		f1=[]
		#分装k fold的下标并返回
		for train_index_0,test_index_0 in kf0:
			f0.append([train_index_0.tolist(),test_index_0.tolist()])
		for train_index_1,test_index_1 in kf1:
			f1.append([train_index_1.tolist(),test_index_1.tolist()])
		return f0,f1


	def level_train(self,clf,level,name,X_0,X_1,uid_0,uid_1):
		"""
		:type clf: scikit-learn classifier or regressor scikit-learn分类器或回归器
		:type level: str 训练第几层
		:type name: str 分类器命名
		:type X_0: numpy.array 类别0特征矩阵
		:type X_1: numpy.array 类型1特征矩阵
		:type uid_0: List 类别0 uid
		:type uid_1: List 类别1 uid
		:层次训练方法，将正负类数据分别分为n folds，然后用(n-1) folds作为训练集
		:用1 fold作为测试集，循环训练，得到一维特征的数据输出
		"""
		n_folds=self.config.n_folds
		f0,f1=self.fold(len(X_0),len(X_1),n_folds) #获得n folds 下标

		predicts=[]
		test_uids=[]
		scores=[]
		part_uids=[]

		for i in range(n_folds):
			train_index_0,test_index_0=f0[i][0],f0[i][1]
			train_index_1,test_index_1=f1[i][0],f1[i][1]

			train_1=X_1[train_index_1]
			test_1=X_1[test_index_1]

			train_0=X_0[train_index_0]
			test_0=X_0[test_index_0]

			test_uid_1=uid_1[test_index_1]
			test_uid_0=uid_0[test_index_0]

			#类的标签直接获得
			y_train=np.hstack((np.ones(len(train_1)),np.zeros(len(train_0))))
			y_test=np.hstack((np.ones(len(test_1)),np.zeros(len(test_0))))

			#测试fold中的uid合并
			test_uid=np.hstack((test_uid_1,test_uid_0))

			#训练集和测试集
			x_train=np.vstack((train_1,train_0))
			x_test=np.vstack((test_1,test_0))

			clf.fit(x_train,y_train)
			try:
				#分类器输出probability
				y_pred=clf.predict_proba(x_test)
				y_pred=y_pred[:,1]
			except:	
				#回归器直接输出预测值
				y_pred=clf.predict(x_test)

			auc_score=metrics.roc_auc_score(y_test,y_pred)
			predicts.extend((y_pred).tolist())
			test_uids.extend(test_uid.tolist())

			print auc_score
			scores.append(auc_score)

		#self.output_level_train(predicts,test_uids,scores,level,name)

		print name+" average scores:",np.mean(scores)

	def xgb_level_train(self,level,name,X_0,X_1,uid_0,uid_1,params,round):
		n_folds=self.config.n_folds
		f0,f1=self.fold(len(X_0),len(X_1),n_folds)

		predicts=[]
		test_uids=[]
		scores=[]
		part_uids=[]

		for i in range(n_folds):
			train_index_0,test_index_0=f0[i][0],f0[i][1]
			train_index_1,test_index_1=f1[i][0],f1[i][1]

			train_1=X_1[train_index_1]
			test_1=X_1[test_index_1]

			train_0=X_0[train_index_0]
			test_0=X_0[test_index_0]

			test_uid_1=uid_1[test_index_1]
			test_uid_0=uid_0[test_index_0]

			train_1=np.vstack((train_1,train_1))

			y_train=np.hstack((np.ones(len(train_1)),np.zeros(len(train_0))))		
			y_test=np.hstack((np.ones(len(test_1)),np.zeros(len(test_0))))

			test_uid=np.hstack((test_uid_1,test_uid_0))

			x_train=np.vstack((train_1,train_0))
			x_test=np.vstack((test_1,test_0))

			dtest=xgb.DMatrix(x_test)
			dtrain=xgb.DMatrix(x_train,label=y_train)
			watchlist=[(dtrain,'train')]

			model=xgb.train(params,dtrain,num_boost_round=round,evals=watchlist,verbose_eval=False)
			y_pred=model.predict(dtest)

			auc_score=metrics.roc_auc_score(y_test,y_pred)
			predicts.extend((y_pred).tolist())
			test_uids.extend(test_uid.tolist())

			#找出部分难分样本
			#part_uid==self.part_train_data(y_test, y_pred, test_uid)
			#part_uids.extend(part_uid)
			print auc_score
			scores.append(auc_score)
			#return

		#self.output_level_train(predicts,test_uids,scores,level,name)
		self.output_part_uid(part_uids,level,name)
		print name+" average scores:",np.mean(scores)


	def part_train_data(self,y_test,y_pred,test_uid):
		sorted_index=sorted(range(len(y_pred)),key=lambda k:y_pred[k],reverse=False)
		y_pred=y_pred[sorted_index]
		y_test=y_test[sorted_index]
		test_uid=test_uid[sorted_index]
		print len(y_test)
		j=0
		k=0
		a=1000
		b=1500
		for i in range(len(y_test[a:b])):
			if y_test[i]==1:
				j+=1
			if i<10 and y_test[i]==0:
				k+=1
				y_pred[i]=0
				#print y_test[i],' ',y_pred[i],' ',test_uid[i],' ',i
		print 'j:',j,' k:',k
		print 'part auc:',metrics.roc_auc_score(y_test[a:b],y_pred[a:b])
		#print 'other auc:',metrics.roc_auc_score(np.hstack((y_test[:a],y_test[b:])),np.hstack((y_pred[:a],y_pred[b:])))
		return test_uid[a:b]

	def output_part_uid(self,part_uids,level,name):
		f1=open(self.config.path_train+level+'/'+name+'_part_uid.csv','wb')
		part_uids=set(part_uids)
		for uid in part_uids:
			f1.write(str(uid)+'\n')
		f1.close()

	def output_level_train(self,predicts,test_uids,scores,level,name):	
		f1=open(self.config.path_train+level+'/'+name+'.csv','wb')
		f2=open(self.config.path_train+level+'/'+name+'_score.csv','wb')
		for i in range(len(test_uids)):
			f1.write(test_uids[i]+","+str(predicts[i])+"\n")

		for score in scores:
			f2.write(str(score)+"\n")

		f1.close()
		f2.close()

	def level_predict(self,clf,level,name,X_0,X_1,predict_X,predict_uid):
		start=datetime.now()
		x_train=np.vstack((X_1,X_0))
		y_train=np.hstack((np.ones(len(X_1)),np.zeros(len(X_0))))

		clf.fit(x_train,y_train)
		try:
			pred_result=clf.predict_proba(predict_X)
			self.output_level_predict(pred_result[:,1],predict_uid,level,name)
		except:
			pred_result=clf.predict(predict_X)
			self.output_level_predict(pred_result,predict_uid,level,name)
		
		end=datetime.now()
		print "finish predict:"+name+" Run time:"+str(float((end-start).seconds)/60.0)+"min / "+str(float((end-start).seconds))+"s"

	def xgb_predict(self,level,name,X_0,X_1,predict_X,predict_uid,params,round):
		start=datetime.now()
		x_train=np.vstack((X_1,X_0))
		y_train=np.hstack((np.ones(len(X_1)),np.zeros(len(X_0))))
		dtrain=xgb.DMatrix(x_train,label=y_train)
		watchlist=[(dtrain,'train')]
		model=xgb.train(params,dtrain,num_boost_round=round,evals=watchlist,verbose_eval=False)

		dpredict=xgb.DMatrix(predict_X)
		predict_result=model.predict(dpredict)
		self.output_level_predict(predict_result,predict_uid,level,name)
		end=datetime.now()
		print "finish predict:"+name+" Run time:"+str(float((end-start).seconds)/60.0)+"min / "+str(float((end-start).seconds))+"s"

	def output_level_predict(self,predicts,test_uids,level,name):	
		f1=open(self.config.path_predict+level+'/'+name+'.csv','wb')
		f1.write('"uid","score"\n')
		for i in range(len(test_uids)):
			f1.write(str(test_uids[i])+","+str(predicts[i])+"\n")
		f1.close()