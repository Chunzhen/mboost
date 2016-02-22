#! /usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import time

from config import Config

import threading

import preprocessing
import load_data
import load_train_data
import load_predict_data
import mboost

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

from level_train_thread import Level_train_thread
from level_predict_thread import Level_predict_thread
from xgb_level_train_thread import Xgb_level_train_thread
from xgb_level_predict_thread import Xgb_level_predict_thread

#数据预处理
def scale_wrapper():
	scales=['log','log_move','standard','normalize','min_max','median']
	threads=[]
	for x in scales:
		config_instance=Config(x)
		preprocessing_instance=preprocessing.Preprocessing(config_instance)
		threads.append(threading.Thread(target=preprocessing_instance.scale_X))

	for t in threads:
		t.start()

	for t in threads:
		t.join()
		
#训练
def level_one_wrapper():
	ftype='standard'
	config_instance=Config(ftype)
	load_data_instance=load_data.Load_data(config_instance)
	#X_0,X_1,uid_0,uid_1=load_data_instance.train_xy()
	#本地训练，取出一份作本地验证
	X_0,test_X_0,X_1,test_X_1,uid_0,test_uid_0,uid_1,test_uid_1=load_data_instance.train_test_xy(1)

	threads=[]
	threads.append(Level_train_thread(config_instance,LogisticRegression(solver='sag'),'level_one',ftype+'_lr_sag',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,LogisticRegression(solver='newton-cg'),'level_one',ftype+'_lr_newton',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,LogisticRegression(solver='lbfgs'),'level_one',ftype+'_lr_lbfgs',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,LogisticRegression(solver='liblinear'),'level_one',ftype+'_lr_liblinear',X_0,X_1,uid_0,uid_1))
	
	threads.append(Level_train_thread(config_instance,RandomForestClassifier(n_estimators=100,max_depth=8,min_samples_split=9),'level_one',ftype+'_rf100',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,RandomForestClassifier(n_estimators=200,max_depth=8,min_samples_split=9),'level_one',ftype+'_rf200',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,RandomForestClassifier(n_estimators=500,max_depth=8,min_samples_split=9),'level_one',ftype+'_rf500',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,RandomForestClassifier(n_estimators=1000,max_depth=8,min_samples_split=9),'level_one',ftype+'_rf1000',X_0,X_1,uid_0,uid_1))

	threads.append(Level_train_thread(config_instance,GradientBoostingClassifier(n_estimators=20,max_depth=8,min_samples_split=9,learning_rate=0.02,subsample=0.7),'level_one',ftype+'_gbdt20',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,GradientBoostingClassifier(n_estimators=50,max_depth=8,min_samples_split=9,learning_rate=0.02,subsample=0.7),'level_one',ftype+'_gbdt50',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,GradientBoostingClassifier(n_estimators=100,max_depth=8,min_samples_split=9,learning_rate=0.02,subsample=0.7),'level_one',ftype+'_gbdt100',X_0,X_1,uid_0,uid_1))

	threads.append(Level_train_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=8,min_samples_split=9),n_estimators=20,learning_rate=0.02),'level_one',ftype+'_ada20',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=8,min_samples_split=9),n_estimators=50,learning_rate=0.02),'level_one',ftype+'_ada50',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=8,min_samples_split=9),n_estimators=100,learning_rate=0.02),'level_one',ftype+'_ada100',X_0,X_1,uid_0,uid_1))

	params={
	    'booster':'gbtree',
	    'objective': 'binary:logistic',
	   	'scale_pos_weight':13458.0/(1400.0),
	    'eval_metric': 'auc',
	    'gamma':0,
	    'max_depth':8,
	    'lambda':700,
	    'subsample':0.7,
	    'colsample_bytree':0.3,
	    'min_child_weight':5,
	    'eta': 0.02,
	    'seed':1,
	    'nthread':8
	    }
	threads.append(Xgb_level_train_thread(config_instance,'level_one',ftype+'_xgb2000',X_0,X_1,uid_0,uid_1,params,2000))
	threads.append(Xgb_level_train_thread(config_instance,'level_one',ftype+'_xgb2500',X_0,X_1,uid_0,uid_1,params,2500))
	params={
	    'booster':'gbtree',
	    'objective': 'binary:logistic',
	   	'scale_pos_weight':13458.0/(1300.0),
	    'eval_metric': 'auc',
	    'gamma':0,
	    'max_depth':8,
	    'lambda':700,
	    'subsample':0.7,
	    'colsample_bytree':0.3,
	    'min_child_weight':5,
	    'eta': 0.02,
	    'seed':1,
	    'nthread':8
	    }
	threads.append(Xgb_level_train_thread(config_instance,'level_one',ftype+'_xgb2000_2',X_0,X_1,uid_0,uid_1,params,2000))
	threads.append(Xgb_level_train_thread(config_instance,'level_one',ftype+'_xgb2500_2',X_0,X_1,uid_0,uid_1,params,2500))

	for thread in threads:
		thread.run()

	# for thread in threads:
	# 	thread.join()

def level_two_wrapper():
	ftype='log_move'
	config_instance=Config(ftype)
	level='level_two'
	clf_name=[
		ftype+'_lr_sag',
		# ftype+'_lr_newton',
		# ftype+'_lr_lbfgs',
		# ftype+'_lr_liblinear',
		# ftype+'_rf100',
		# ftype+'_rf200',
		# ftype+'_rf500',
		ftype+'_rf1000',
		# ftype+'_gbdt20',
		# ftype+'_gbdt50',
		ftype+'_gbdt100',
		ftype+'_ada20',
		ftype+'_ada50',
		ftype+'_ada100',
		ftype+'_xgb2000',
		ftype+'_xgb2500',
		ftype+'_xgb2000_2',
		ftype+'_xgb2500_2'
	]
	# ftype='standard'
	# clf_name2=[
	# 	ftype+'_lr_sag',
	# 	# ftype+'_lr_newton',
	# 	# ftype+'_lr_lbfgs',
	# 	# ftype+'_lr_liblinear',
	# 	# ftype+'_rf100',
	# 	# ftype+'_rf200',
	# 	# ftype+'_rf500',
	# 	# ftype+'_rf1000',
	# 	# ftype+'_gbdt20',
	# 	# ftype+'_gbdt50',
	# 	# ftype+'_gbdt100',
	# 	# ftype+'_ada20',
	# 	# ftype+'_ada50',
	# 	# ftype+'_ada100',
	# 	ftype+'_xgb2000',
	# 	ftype+'_xgb2500',
	# 	ftype+'_xgb2000_2',
	# 	ftype+'_xgb2500_2'
	# ]
	# clf_name.extend(clf_name2)
	#ftype='log'

	load_data_instance=load_train_data.Load_train_data(config_instance,'level_one',clf_name)
	X_0,X_1,uid_0,uid_1=load_data_instance.level_data()

	threads=[]

	threads.append(Level_train_thread(config_instance,LogisticRegression(solver='sag'),level,ftype+'_lr_sag2',X_0,X_1,uid_0,uid_1))
	# threads.append(Level_train_thread(config_instance,LogisticRegression(solver='newton-cg'),level,ftype+'_lr_newton',X_0,X_1,uid_0,uid_1))
	# threads.append(Level_train_thread(config_instance,LogisticRegression(solver='lbfgs'),level,ftype+'_lr_lbfgs',X_0,X_1,uid_0,uid_1))
	# threads.append(Level_train_thread(config_instance,LogisticRegression(solver='liblinear'),level,ftype+'_lr_liblinear',X_0,X_1,uid_0,uid_1))
	
	#threads.append(Level_train_thread(config_instance,RandomForestClassifier(n_estimators=100,max_depth=2,min_samples_split=10),level,ftype+'_rf100_2',X_0,X_1,uid_0,uid_1))
	# threads.append(Level_train_thread(config_instance,RandomForestClassifier(n_estimators=200,max_depth=3,min_samples_split=10),level,ftype+'_rf200_2',X_0,X_1,uid_0,uid_1))
	# threads.append(Level_train_thread(config_instance,RandomForestClassifier(n_estimators=500,max_depth=3,min_samples_split=10),level,ftype+'_rf500',X_0,X_1,uid_0,uid_1))
	# threads.append(Level_train_thread(config_instance,RandomForestClassifier(n_estimators=1000,max_depth=3,min_samples_split=10),level,ftype+'_rf1000',X_0,X_1,uid_0,uid_1))

	#threads.append(Level_train_thread(config_instance,GradientBoostingClassifier(n_estimators=200,max_depth=3,min_samples_split=15,learning_rate=0.005,subsample=0.7),level,ftype+'_gbdt20_1',X_0,X_1,uid_0,uid_1))
	# threads.append(Level_train_thread(config_instance,GradientBoostingClassifier(n_estimators=50,max_depth=3,min_samples_split=15,learning_rate=0.01,subsample=0.7),level,ftype+'_gbdt50',X_0,X_1,uid_0,uid_1))
	# threads.append(Level_train_thread(config_instance,GradientBoostingClassifier(n_estimators=100,max_depth=3,min_samples_split=15,learning_rate=0.01,subsample=0.7),level,ftype+'_gbdt100',X_0,X_1,uid_0,uid_1))

	#threads.append(Level_train_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=3,min_samples_split=10),n_estimators=20,learning_rate=0.001),level,ftype+'_ada20_222',X_0,X_1,uid_0,uid_1))
	# threads.append(Level_train_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=3,min_samples_split=10),n_estimators=50,learning_rate=0.02),level,ftype+'_ada50',X_0,X_1,uid_0,uid_1))
	# threads.append(Level_train_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=3,min_samples_split=10),n_estimators=100,learning_rate=0.02),level,ftype+'_ada100',X_0,X_1,uid_0,uid_1))

	# params={
	#     'booster':'gbtree',
	#     'objective': 'binary:logistic',
	#    	'scale_pos_weight':13458.0/(1300.0),
	#     'eval_metric': 'auc',
	#     'gamma':15,
	#     'max_depth':3,
	#     'lambda':600,
	#     'subsample':0.40,
	#     'colsample_bytree':0.3,
	#     'min_child_weight':10,
	#     'eta': 0.002,#0.0005
	#     'seed':1,
	#     'nthread':8
	#     }
	# threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb2000',X_0,X_1,uid_0,uid_1,params,3000))
	# threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb2500',X_0,X_1,uid_0,uid_1,params,2500))
	# params={
	#     'booster':'gbtree',
	#     'objective': 'binary:logistic',
	#    	'scale_pos_weight':13458.0/(1300.0),
	#     'eval_metric': 'auc',
	#     'gamma':0,
	#     'max_depth':3,
	#     'lambda':700,
	#     'subsample':0.9,
	#     'colsample_bytree':0.3,
	#     'min_child_weight':5,
	#     'eta': 0.0005,
	#     'seed':1,
	#     'nthread':8
	#     }
	# threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb2000_2',X_0,X_1,uid_0,uid_1,params,2000))
	# threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb2500_2',X_0,X_1,uid_0,uid_1,params,2500))
	for thread in threads:
		thread.run()

def level_three_wrapper():
	ftype='log'
	config_instance=Config(ftype)
	level='level_three'
	clf_name=[
		'log_move_lr_sag',
		'log_move_lr_newton',
		'log_move_lr_lbfgs',
		'log_move_lr_liblinear',
		'log_move_rf100',
		'log_move_rf200',
		'log_move_rf500',
		'log_move_rf1000',
		'log_move_gbdt20',
		'log_move_gbdt50',
		'log_move_gbdt100',
		'log_move_ada20',
		'log_move_ada50',
		'log_move_ada100',
		'log_move_xgb2000',
		'log_move_xgb2500',
		'log_move_xgb2000_2',
		'log_move_xgb2500_2'
	]
	load_data_instance=load_train_data.Load_train_data(config_instance,'level_two',clf_name)
	X_0,X_1,uid_0,uid_1=load_data_instance.level_data()

	threads=[]
	threads.append(Level_train_thread(config_instance,LogisticRegression(solver='sag'),level,ftype+'_lr_sag',X_0,X_1,uid_0,uid_1))
	# threads.append(Level_train_thread(config_instance,LogisticRegression(solver='newton-cg'),level,ftype+'_lr_newton',X_0,X_1,uid_0,uid_1))
	# threads.append(Level_train_thread(config_instance,LogisticRegression(solver='lbfgs'),level,ftype+'_lr_lbfgs',X_0,X_1,uid_0,uid_1))
	# threads.append(Level_train_thread(config_instance,LogisticRegression(solver='liblinear'),level,ftype+'_lr_liblinear',X_0,X_1,uid_0,uid_1))
	
	#threads.append(Level_train_thread(config_instance,RandomForestClassifier(n_estimators=100,max_depth=3,min_samples_split=10),level,ftype+'_rf100',X_0,X_1,uid_0,uid_1))

	# threads.append(Level_train_thread(config_instance,RandomForestClassifier(n_estimators=200,max_depth=3,min_samples_split=10),level,ftype+'_rf200',X_0,X_1,uid_0,uid_1))
	# threads.append(Level_train_thread(config_instance,RandomForestClassifier(n_estimators=500,max_depth=3,min_samples_split=10),level,ftype+'_rf500',X_0,X_1,uid_0,uid_1))
	#threads.append(Level_train_thread(config_instance,RandomForestClassifier(n_estimators=1000,max_depth=3,min_samples_split=10),level,ftype+'_rf1000',X_0,X_1,uid_0,uid_1))

	# threads.append(Level_train_thread(config_instance,GradientBoostingClassifier(n_estimators=20,max_depth=3,min_samples_split=15,learning_rate=0.01,subsample=0.7),level,ftype+'_gbdt20',X_0,X_1,uid_0,uid_1))
	# threads.append(Level_train_thread(config_instance,GradientBoostingClassifier(n_estimators=50,max_depth=3,min_samples_split=15,learning_rate=0.01,subsample=0.7),level,ftype+'_gbdt50',X_0,X_1,uid_0,uid_1))
	#threads.append(Level_train_thread(config_instance,GradientBoostingClassifier(n_estimators=100,max_depth=3,min_samples_split=15,learning_rate=0.01,subsample=0.7),level,ftype+'_gbdt100',X_0,X_1,uid_0,uid_1))

	#threads.append(Level_train_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=2,min_samples_split=10),n_estimators=20,learning_rate=0.02),level,ftype+'_ada20',X_0,X_1,uid_0,uid_1))
	# threads.append(Level_train_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=3,min_samples_split=10),n_estimators=50,learning_rate=0.02),level,ftype+'_ada50',X_0,X_1,uid_0,uid_1))
	# threads.append(Level_train_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=3,min_samples_split=10),n_estimators=100,learning_rate=0.02),level,ftype+'_ada100',X_0,X_1,uid_0,uid_1))

	# params={
	#     'booster':'gbtree',
	#     'objective': 'binary:logistic',
	#    	'scale_pos_weight':13458.0/(1400.0),
	#     'eval_metric': 'auc',
	#     'gamma':0,
	#     'max_depth':3,
	#     'lambda':700,
	#     'subsample':0.9,
	#     'colsample_bytree':0.3,
	#     'min_child_weight':5,
	#     'eta': 0.0005,
	#     'seed':1,
	#     'nthread':8
	#     }
	# threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb2000',X_0,X_1,uid_0,uid_1,params,2000))
	# threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb2500',X_0,X_1,uid_0,uid_1,params,2500))
	# params={
	#     'booster':'gbtree',
	#     'objective': 'binary:logistic',
	#    	'scale_pos_weight':13458.0/(1300.0),
	#     'eval_metric': 'auc',
	#     'gamma':0,
	#     'max_depth':3,
	#     'lambda':700,
	#     'subsample':0.9,
	#     'colsample_bytree':0.3,
	#     'min_child_weight':5,
	#     'eta': 0.0005,
	#     'seed':1,
	#     'nthread':8
	#     }
	# threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb2000_2',X_0,X_1,uid_0,uid_1,params,2000))
	# threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb2500_2',X_0,X_1,uid_0,uid_1,params,2500))

	for thread in threads:
		thread.run()
	pass
#预测

def level_one_predict():
	ftype='log'
	config_instance=Config(ftype)
	level='level_one'
	load_data_instance=load_data.Load_data(config_instance)
	#本地训练，取出一份作本地验证
	X_0,test_X_0,X_1,test_X_1,uid_0,test_uid_0,uid_1,test_uid_1=load_data_instance.train_test_xy(1)
	predict_X=np.vstack((test_X_0,test_X_1))
	predict_uid=np.hstack((test_uid_0,test_uid_1))

	#X_0,X_1,uid_0,uid_1=load_data_instance.train_xy()
	#predict_X,predict_uid=load_data_instance.predict_X()

	threads=[]
	threads.append(Level_predict_thread(config_instance,LogisticRegression(solver='sag'),level,ftype+'_lr_sag',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,LogisticRegression(solver='newton-cg'),level,ftype+'_lr_newton',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,LogisticRegression(solver='lbfgs'),level,ftype+'_lr_lbfgs',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,LogisticRegression(solver='liblinear'),level,ftype+'_lr_liblinear',X_0,X_1,predict_X,predict_uid))
	
	threads.append(Level_predict_thread(config_instance,RandomForestClassifier(n_estimators=100,max_depth=8,min_samples_split=9),level,ftype+'_rf100',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,RandomForestClassifier(n_estimators=200,max_depth=8,min_samples_split=9),level,ftype+'_rf200',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,RandomForestClassifier(n_estimators=500,max_depth=8,min_samples_split=9),level,ftype+'_rf500',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,RandomForestClassifier(n_estimators=1000,max_depth=8,min_samples_split=9),level,ftype+'_rf1000',X_0,X_1,predict_X,predict_uid))

	threads.append(Level_predict_thread(config_instance,GradientBoostingClassifier(n_estimators=20,max_depth=8,min_samples_split=9,learning_rate=0.02,subsample=0.7),level,ftype+'_gbdt20',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,GradientBoostingClassifier(n_estimators=50,max_depth=8,min_samples_split=9,learning_rate=0.02,subsample=0.7),level,ftype+'_gbdt50',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,GradientBoostingClassifier(n_estimators=100,max_depth=8,min_samples_split=9,learning_rate=0.02,subsample=0.7),level,ftype+'_gbdt100',X_0,X_1,predict_X,predict_uid))

	threads.append(Level_predict_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=8,min_samples_split=9),n_estimators=20,learning_rate=0.02),level,ftype+'_ada20',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=8,min_samples_split=9),n_estimators=50,learning_rate=0.02),level,ftype+'_ada50',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=8,min_samples_split=9),n_estimators=100,learning_rate=0.02),level,ftype+'_ada100',X_0,X_1,predict_X,predict_uid))

	params={
	    'booster':'gbtree',
	    'objective': 'binary:logistic',
	   	'scale_pos_weight':13458.0/(1400.0),
	    'eval_metric': 'auc',
	    'gamma':0,
	    'max_depth':8,
	    'lambda':700,
	    'subsample':0.7,
	    'colsample_bytree':0.3,
	    'min_child_weight':5,
	    'eta': 0.02,
	    'seed':1,
	    'nthread':8
	    }
	threads.append(Xgb_level_predict_thread(config_instance,level,ftype+'_xgb2000',X_0,X_1,predict_X,predict_uid,params,2000))
	threads.append(Xgb_level_predict_thread(config_instance,level,ftype+'_xgb2500',X_0,X_1,predict_X,predict_uid,params,2500))
	params={
	    'booster':'gbtree',
	    'objective': 'binary:logistic',
	   	'scale_pos_weight':13458.0/(1300.0),
	    'eval_metric': 'auc',
	    'gamma':0,
	    'max_depth':8,
	    'lambda':700,
	    'subsample':0.7,
	    'colsample_bytree':0.3,
	    'min_child_weight':5,
	    'eta': 0.02,
	    'seed':1,
	    'nthread':8
	    }
	threads.append(Xgb_level_predict_thread(config_instance,level,ftype+'_xgb2000_2',X_0,X_1,predict_X,predict_uid,params,2000))
	threads.append(Xgb_level_predict_thread(config_instance,level,ftype+'_xgb2500_2',X_0,X_1,predict_X,predict_uid,params,2500))

	for thread in threads:
		thread.run()

	pass

def level_two_predict():
	config_instance=Config('log_move')
	level='level_two'
	clf_name=[
		'log_move_lr_sag',
		# 'log_move_lr_newton',
		# 'log_move_lr_lbfgs',
		# 'log_move_lr_liblinear',
		# 'log_move_rf100',
		# 'log_move_rf200',
		# 'log_move_rf500',
		'log_move_rf1000',
		# 'log_move_gbdt20',
		# 'log_move_gbdt50',
		'log_move_gbdt100',
		'log_move_ada20',
		'log_move_ada50',
		'log_move_ada100',
		'log_move_xgb2000',
		'log_move_xgb2500',
		'log_move_xgb2000_2',
		'log_move_xgb2500_2'
	]
	load_data_instance=load_train_data.Load_train_data(config_instance,'level_one',clf_name)
	predict_data_instance=load_predict_data.Load_predict_data(config_instance,'level_one',clf_name)
	X_0,X_1,uid_0,uid_1=load_data_instance.level_data()
	predict_X,predict_uid=predict_data_instance.level_data()

	threads=[]
	#threads.append(Level_predict_thread(config_instance,LogisticRegression(solver='sag'),level,'log_move_lr_sag',X_0,X_1,predict_X,predict_uid))
	# threads.append(Level_predict_thread(config_instance,LogisticRegression(solver='newton-cg'),level,'log_move_lr_newton',X_0,X_1,predict_X,predict_uid))
	# threads.append(Level_predict_thread(config_instance,LogisticRegression(solver='lbfgs'),level,'log_move_lr_lbfgs',X_0,X_1,predict_X,predict_uid))
	# threads.append(Level_predict_thread(config_instance,LogisticRegression(solver='liblinear'),level,'log_move_lr_liblinear',X_0,X_1,predict_X,predict_uid))

	# threads.append(Level_predict_thread(config_instance,RandomForestClassifier(n_estimators=100,max_depth=3,min_samples_split=10),level,'log_move_rf100',X_0,X_1,predict_X,predict_uid))
	# threads.append(Level_predict_thread(config_instance,RandomForestClassifier(n_estimators=200,max_depth=3,min_samples_split=10),level,'log_move_rf200',X_0,X_1,predict_X,predict_uid))
	# threads.append(Level_predict_thread(config_instance,RandomForestClassifier(n_estimators=500,max_depth=3,min_samples_split=10),level,'log_move_rf500',X_0,X_1,predict_X,predict_uid))
	# threads.append(Level_predict_thread(config_instance,RandomForestClassifier(n_estimators=1000,max_depth=3,min_samples_split=10),level,'log_move_rf1000',X_0,X_1,predict_X,predict_uid))

	# threads.append(Level_predict_thread(config_instance,GradientBoostingClassifier(n_estimators=20,max_depth=3,min_samples_split=15,learning_rate=0.01,subsample=0.7),level,'log_move_gbdt20',X_0,X_1,predict_X,predict_uid))
	# threads.append(Level_predict_thread(config_instance,GradientBoostingClassifier(n_estimators=50,max_depth=3,min_samples_split=15,learning_rate=0.01,subsample=0.7),level,'log_move_gbdt50',X_0,X_1,predict_X,predict_uid))
	# threads.append(Level_predict_thread(config_instance,GradientBoostingClassifier(n_estimators=100,max_depth=3,min_samples_split=15,learning_rate=0.01,subsample=0.7),level,'log_move_gbdt100',X_0,X_1,predict_X,predict_uid))

	# threads.append(Level_predict_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=3,min_samples_split=10),n_estimators=20,learning_rate=0.02),level,'log_move_ada20',X_0,X_1,predict_X,predict_uid))
	# threads.append(Level_predict_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=3,min_samples_split=10),n_estimators=50,learning_rate=0.02),level,'log_move_ada50',X_0,X_1,predict_X,predict_uid))
	# threads.append(Level_predict_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=3,min_samples_split=10),n_estimators=100,learning_rate=0.02),level,'log_move_ada100',X_0,X_1,predict_X,predict_uid))

	params={
	    'booster':'gbtree',
	    'objective': 'binary:logistic',
	   	'scale_pos_weight':13458.0/(1400.0),
	    'eval_metric': 'auc',
	    'gamma':15,
	    'max_depth':3,
	    'lambda':600,
	    'subsample':0.40,
	    'colsample_bytree':0.3,
	    'min_child_weight':10,
	    'eta': 0.002,#0.0005
	    'seed':1,
	    'nthread':8
	    }

	threads.append(Xgb_level_predict_thread(config_instance,level,'log_move_xgb2000_222',X_0,X_1,predict_X,predict_uid,params,2000))
	# threads.append(Xgb_level_predict_thread(config_instance,level,'log_move_xgb2500',X_0,X_1,predict_X,predict_uid,params,2500))
	# params={
	#     'booster':'gbtree',
	#     'objective': 'binary:logistic',
	#    	'scale_pos_weight':13458.0/(1300.0),
	#     'eval_metric': 'auc',
	#     'gamma':0,
	#     'max_depth':3,
	#     'lambda':700,
	#     'subsample':0.9,
	#     'colsample_bytree':0.3,
	#     'min_child_weight':5,
	#     'eta': 0.0005,
	#     'seed':1,
	#     'nthread':8
	#     }
	# threads.append(Xgb_level_predict_thread(config_instance,level,'log_move_xgb2000_2',X_0,X_1,predict_X,predict_uid,params,2000))
	# threads.append(Xgb_level_predict_thread(config_instance,level,'log_move_xgb2500_2',X_0,X_1,predict_X,predict_uid,params,2500))
	for thread in threads:
		thread.run()
	pass

def level_three_predict():
	config_instance=Config('log_move')
	level='level_three'
	clf_name=[
		'log_move_lr_sag',
		'log_move_lr_newton',
		'log_move_lr_lbfgs',
		'log_move_lr_liblinear',
		'log_move_rf100',
		'log_move_rf200',
		'log_move_rf500',
		'log_move_rf1000',
		'log_move_gbdt20',
		'log_move_gbdt50',
		'log_move_gbdt100',
		'log_move_ada20',
		'log_move_ada50',
		'log_move_ada100',
		'log_move_xgb2000',
		'log_move_xgb2500',
		'log_move_xgb2000_2',
		'log_move_xgb2500_2'
	]
	load_data_instance=load_train_data.Load_train_data(config_instance,'level_two',clf_name)
	predict_data_instance=load_predict_data.Load_predict_data(config_instance,'level_two',clf_name)
	X_0,X_1,uid_0,uid_1=load_data_instance.level_data()
	predict_X,predict_uid=predict_data_instance.level_data()

	threads=[]
	threads.append(Level_predict_thread(config_instance,LogisticRegression(solver='sag'),level,'log_move_lr_sag',X_0,X_1,predict_X,predict_uid))
	# threads.append(Level_predict_thread(config_instance,LogisticRegression(solver='newton-cg'),level,'log_move_lr_newton',X_0,X_1,predict_X,predict_uid))
	# threads.append(Level_predict_thread(config_instance,LogisticRegression(solver='lbfgs'),level,'log_move_lr_lbfgs',X_0,X_1,predict_X,predict_uid))
	# threads.append(Level_predict_thread(config_instance,LogisticRegression(solver='liblinear'),level,'log_move_lr_liblinear',X_0,X_1,predict_X,predict_uid))

	# threads.append(Level_predict_thread(config_instance,RandomForestClassifier(n_estimators=100,max_depth=3,min_samples_split=10),level,'log_move_rf100',X_0,X_1,predict_X,predict_uid))
	# threads.append(Level_predict_thread(config_instance,RandomForestClassifier(n_estimators=200,max_depth=3,min_samples_split=10),level,'log_move_rf200',X_0,X_1,predict_X,predict_uid))
	# threads.append(Level_predict_thread(config_instance,RandomForestClassifier(n_estimators=500,max_depth=3,min_samples_split=10),level,'log_move_rf500',X_0,X_1,predict_X,predict_uid))
	# threads.append(Level_predict_thread(config_instance,RandomForestClassifier(n_estimators=1000,max_depth=3,min_samples_split=10),level,'log_move_rf1000',X_0,X_1,predict_X,predict_uid))

	# threads.append(Level_predict_thread(config_instance,GradientBoostingClassifier(n_estimators=20,max_depth=3,min_samples_split=15,learning_rate=0.01,subsample=0.7),level,'log_move_gbdt20',X_0,X_1,predict_X,predict_uid))
	# threads.append(Level_predict_thread(config_instance,GradientBoostingClassifier(n_estimators=50,max_depth=3,min_samples_split=15,learning_rate=0.01,subsample=0.7),level,'log_move_gbdt50',X_0,X_1,predict_X,predict_uid))
	# threads.append(Level_predict_thread(config_instance,GradientBoostingClassifier(n_estimators=100,max_depth=3,min_samples_split=15,learning_rate=0.01,subsample=0.7),level,'log_move_gbdt100',X_0,X_1,predict_X,predict_uid))

	# threads.append(Level_predict_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=3,min_samples_split=10),n_estimators=20,learning_rate=0.02),level,'log_move_ada20',X_0,X_1,predict_X,predict_uid))
	# threads.append(Level_predict_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=3,min_samples_split=10),n_estimators=50,learning_rate=0.02),level,'log_move_ada50',X_0,X_1,predict_X,predict_uid))
	# threads.append(Level_predict_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=3,min_samples_split=10),n_estimators=100,learning_rate=0.02),level,'log_move_ada100',X_0,X_1,predict_X,predict_uid))

	# params={
	#     'booster':'gbtree',
	#     'objective': 'binary:logistic',
	#    	'scale_pos_weight':13458.0/(1400.0),
	#     'eval_metric': 'auc',
	#     'gamma':0,
	#     'max_depth':3,
	#     'lambda':700,
	#     'subsample':0.9,
	#     'colsample_bytree':0.3,
	#     'min_child_weight':5,
	#     'eta': 0.0005,
	#     'seed':1,
	#     'nthread':8
	#     }
	# threads.append(Xgb_level_predict_thread(config_instance,level,'log_move_xgb2000',X_0,X_1,predict_X,predict_uid,params,2000))
	# threads.append(Xgb_level_predict_thread(config_instance,level,'log_move_xgb2500',X_0,X_1,predict_X,predict_uid,params,2500))
	# params={
	#     'booster':'gbtree',
	#     'objective': 'binary:logistic',
	#    	'scale_pos_weight':13458.0/(1300.0),
	#     'eval_metric': 'auc',
	#     'gamma':0,
	#     'max_depth':3,
	#     'lambda':700,
	#     'subsample':0.9,
	#     'colsample_bytree':0.3,
	#     'min_child_weight':5,
	#     'eta': 0.0005,
	#     'seed':1,
	#     'nthread':8
	#     }
	# threads.append(Xgb_level_predict_thread(config_instance,level,'log_move_xgb2000_2',X_0,X_1,predict_X,predict_uid,params,2000))
	# threads.append(Xgb_level_predict_thread(config_instance,level,'log_move_xgb2500_2',X_0,X_1,predict_X,predict_uid,params,2500))
	for thread in threads:
		thread.run()
	pass

def main():
	#level_one_wrapper()
	level_two_wrapper()
	#level_three_wrapper()
	#level_one_predict()
	#level_two_predict()
	#level_three_predict()

if __name__ == '__main__':
	reload(sys)
	sys.setdefaultencoding('utf8')
	start=datetime.now()
	main()
	end=datetime.now()
	print "All Run time:"+str(float((end-start).seconds)/60.0)+"min / "+str(float((end-start).seconds))+"s"
