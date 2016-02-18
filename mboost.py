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
	def __init__(self,config):
		self.config=config
		pass

	def fold(self,len_0,len_1,n_folds):
		random_state=self.config.fold_random_state
		kf0=KFold(n=len_0, n_folds=n_folds, shuffle=True,random_state=random_state)
		kf1=KFold(n=len_1,n_folds=n_folds, shuffle=True,random_state=random_state)
		f0=[]
		f1=[]
		for train_index_0,test_index_0 in kf0:
			f0.append([train_index_0.tolist(),test_index_0.tolist()])
		for train_index_1,test_index_1 in kf1:
			f1.append([train_index_1.tolist(),test_index_1.tolist()])
		return f0,f1


	def level_train(self,clf,level,name,X_0,X_1,uid_0,uid_1):
		n_folds=self.config.n_folds
		f0,f1=self.fold(len(X_0),len(X_1),n_folds)

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

			clf.fit(x_train,y_train)

			y_pred=clf.predict_proba(x_test)

			auc_score=metrics.roc_auc_score(y_test,y_pred[:,1])
			predicts.extend((y_pred[:,1]).tolist())
			test_uids.extend(test_uid.tolist())

			#print auc_score
			scores.append(auc_score)

		self.output_level_train(predicts,test_uids,scores,level,name)
		print name+" average scores:",np.mean(scores)

	def xgb_level_train(self,level,name,X_0,X_1,uid_0,uid_1,params,round):
		n_folds=self.config.n_folds
		f0,f1=self.fold(len(X_0),len(X_1),n_folds)

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

			dtest=xgb.DMatrix(x_test)
			dtrain=xgb.DMatrix(x_train,label=y_train)
			watchlist=[(dtrain,'train')]

			model=xgb.train(params,dtrain,num_boost_round=round,evals=watchlist,verbose_eval=False)
			y_pred=model.predict(dtest)

			auc_score=metrics.roc_auc_score(y_test,y_pred)
			predicts.extend((y_pred).tolist())
			test_uids.extend(test_uid.tolist())

			#print auc_score
			scores.append(auc_score)

		self.output_level_train(predicts,test_uids,scores,level,name)
		#print name+" average scores:",np.mean(scores)

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

		pred_result=clf.predict_proba(predict_X)
		self.output_level_predict(pred_result[:,1],predict_uid,level,name)
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