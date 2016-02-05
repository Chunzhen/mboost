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
import mboost

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

import logging

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='predict_level_one.log',
                filemode='a+')

logging.info("**********************************New Train**********************************")

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
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

class Level_train_thread(object):
	def __init__(self,config,clf,level,name,X_0,X_1,uid_0,uid_1):
		#threading.Thread.__init__(self)
		self.config=config
		self.clf=clf
		self.level=level
		self.name=name
		self.X_0=X_0
		self.X_1=X_1
		self.uid_0=uid_0
		self.uid_1=uid_1

	def run(self):
		logging.info('Begin train '+self.name)
		start=datetime.now()
		boost_instance=mboost.Mboost(self.config)
		boost_instance.level_train(self.clf,self.level,self.name,self.X_0,self.X_1,self.uid_0,self.uid_1)
		end=datetime.now()
		logging.info('End train '+self.name+", cost time: "+str(float((end-start).seconds)/60.0)+"min / "+str(float((end-start).seconds))+"s")

class Xgb_level_train_thread(object):
	def __init__(self,config,level,name,X_0,X_1,uid_0,uid_1,params,round):
		#threading.Thread.__init__(self)
		self.config=config
		self.level=level
		self.name=name
		self.X_0=X_0
		self.X_1=X_1
		self.uid_0=uid_0
		self.uid_1=uid_1
		self.params=params
		self.round=round

	def run(self):
		logging.info('Begin train '+self.name)
		start=datetime.now()
		boost_instance=mboost.Mboost(self.config)
		boost_instance.xgb_level_train(self.level,self.name,self.X_0,self.X_1,self.uid_0,self.uid_1,self.params,self.round)
		end=datetime.now()
		logging.info('End train '+self.name+", cost time: "+str(float((end-start).seconds)/60.0)+"min / "+str(float((end-start).seconds))+"s")

class Level_predict_thread(object):
	def __init__(self,config,clf,level,name,X_0,X_1,predict_X,predict_uid):
		#threading.Thread.__init__(self)
		self.config=config
		self.clf=clf
		self.level=level
		self.name=name
		self.X_0=X_0
		self.X_1=X_1
		self.predict_X=predict_X
		self.predict_uid=predict_uid

	def run(self):
		logging.info('Begin train '+self.name)
		start=datetime.now()
		boost_instance=mboost.Mboost(self.config)
		boost_instance.level_predict(self.clf,self.level,self.name,self.X_0,self.X_1,self.predict_X,self.predict_uid)
		end=datetime.now()
		logging.info('End train '+self.name+", cost time: "+str(float((end-start).seconds)/60.0)+"min / "+str(float((end-start).seconds))+"s")

class Xgb_level_predict_thread(object):
	def __init__(self,config,level,name,X_0,X_1,predict_X,predict_uid,params,round):
		#threading.Thread.__init__(self)
		self.config=config
		self.level=level
		self.name=name
		self.X_0=X_0
		self.X_1=X_1
		self.predict_X=predict_X
		self.predict_uid=predict_uid
		self.params=params
		self.round=round

	def run(self):
		logging.info('Begin train '+self.name)
		start=datetime.now()
		boost_instance=mboost.Mboost(self.config)
		boost_instance.xgb_predict(self.level,self.name,self.X_0,self.X_1,self.predict_X,self.predict_uid,self.params,self.round)
		end=datetime.now()
		logging.info('End train '+self.name+", cost time: "+str(float((end-start).seconds)/60.0)+"min / "+str(float((end-start).seconds))+"s")

		
#训练
def level_one_wrapper():
	config_instance=Config('log_move')
	load_data_instance=load_data.Load_data(config_instance)
	X_0,X_1,uid_0,uid_1=load_data_instance.train_xy()
	
	threads=[]
	# threads.append(Level_train_thread(config_instance,LogisticRegression(solver='sag'),'level_one','log_move_lr_sag',X_0,X_1,uid_0,uid_1))
	# threads.append(Level_train_thread(config_instance,LogisticRegression(solver='newton-cg'),'level_one','log_move_lr_newton',X_0,X_1,uid_0,uid_1))
	# threads.append(Level_train_thread(config_instance,LogisticRegression(solver='lbfgs'),'level_one','log_move_lr_lbfgs',X_0,X_1,uid_0,uid_1))
	# threads.append(Level_train_thread(config_instance,LogisticRegression(solver='liblinear'),'level_one','log_move_lr_liblinear',X_0,X_1,uid_0,uid_1))
	
	#threads.append(Level_train_thread(config_instance,RandomForestClassifier(n_estimators=100,max_depth=8,min_samples_split=9),'level_one','log_move_rf100',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,RandomForestClassifier(n_estimators=200,max_depth=8,min_samples_split=9),'level_one','log_move_rf200',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,RandomForestClassifier(n_estimators=500,max_depth=8,min_samples_split=9),'level_one','log_move_rf500',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,RandomForestClassifier(n_estimators=1000,max_depth=8,min_samples_split=9),'level_one','log_move_rf1000',X_0,X_1,uid_0,uid_1))

	threads.append(Level_train_thread(config_instance,GradientBoostingClassifier(n_estimators=20,max_depth=8,min_samples_split=9,learning_rate=0.02,subsample=0.7),'level_one','log_move_gbdt20',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,GradientBoostingClassifier(n_estimators=50,max_depth=8,min_samples_split=9,learning_rate=0.02,subsample=0.7),'level_one','log_move_gbdt50',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,GradientBoostingClassifier(n_estimators=100,max_depth=8,min_samples_split=9,learning_rate=0.02,subsample=0.7),'level_one','log_move_gbdt100',X_0,X_1,uid_0,uid_1))

	threads.append(Level_train_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=8,min_samples_split=9),n_estimators=20,learning_rate=0.02),'level_one','log_move_ada20',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=8,min_samples_split=9),n_estimators=50,learning_rate=0.02),'level_one','log_move_ada50',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=8,min_samples_split=9),n_estimators=100,learning_rate=0.02),'level_one','log_move_ada100',X_0,X_1,uid_0,uid_1))

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
	threads.append(Xgb_level_train_thread(config_instance,'level_one','log_move_xgb2000',X_0,X_1,uid_0,uid_1,params,2000))
	threads.append(Xgb_level_train_thread(config_instance,'level_one','log_move_xgb2500',X_0,X_1,uid_0,uid_1,params,2500))
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
	threads.append(Xgb_level_train_thread(config_instance,'level_one','log_move_xgb2000_2',X_0,X_1,uid_0,uid_1,params,2000))
	threads.append(Xgb_level_train_thread(config_instance,'level_one','log_move_xgb2500_2',X_0,X_1,uid_0,uid_1,params,2500))

	for thread in threads:
		thread.run()

	# for thread in threads:
	# 	thread.join()

def level_two_wrapper():
	config_instance=Config('log_move')
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
	load_data_instance=load_train_data.Load_train_data(config_instance,'level_one',clf_name)
	X_0,X_1,uid_0,uid_1=load_data_instance.level_data()
	threads=[]
	threads.append(Level_train_thread(config_instance,LogisticRegression(solver='sag'),'level_two','log_move_lr_sag',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,LogisticRegression(solver='newton-cg'),'level_two','log_move_lr_newton',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,LogisticRegression(solver='lbfgs'),'level_two','log_move_lr_lbfgs',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,LogisticRegression(solver='liblinear'),'level_two','log_move_lr_liblinear',X_0,X_1,uid_0,uid_1))
	
	threads.append(Level_train_thread(config_instance,RandomForestClassifier(n_estimators=100,max_depth=4,min_samples_split=9),'level_two','log_move_rf100',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,RandomForestClassifier(n_estimators=200,max_depth=4,min_samples_split=9),'level_two','log_move_rf200',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,RandomForestClassifier(n_estimators=500,max_depth=4,min_samples_split=9),'level_two','log_move_rf500',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,RandomForestClassifier(n_estimators=1000,max_depth=4,min_samples_split=9),'level_two','log_move_rf1000',X_0,X_1,uid_0,uid_1))

	threads.append(Level_train_thread(config_instance,GradientBoostingClassifier(n_estimators=20,max_depth=4,min_samples_split=9,learning_rate=0.02,subsample=0.7),'level_two','log_move_gbdt20',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,GradientBoostingClassifier(n_estimators=50,max_depth=4,min_samples_split=9,learning_rate=0.02,subsample=0.7),'level_two','log_move_gbdt50',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,GradientBoostingClassifier(n_estimators=100,max_depth=4,min_samples_split=9,learning_rate=0.02,subsample=0.7),'level_two','log_move_gbdt100',X_0,X_1,uid_0,uid_1))

	threads.append(Level_train_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=4,min_samples_split=9),n_estimators=20,learning_rate=0.02),'level_two','log_move_ada20',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=4,min_samples_split=9),n_estimators=50,learning_rate=0.02),'level_two','log_move_ada50',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=4,min_samples_split=9),n_estimators=100,learning_rate=0.02),'level_two','log_move_ada100',X_0,X_1,uid_0,uid_1))

	params={
	    'booster':'gbtree',
	    'objective': 'binary:logistic',
	   	'scale_pos_weight':13458.0/(1400.0),
	    'eval_metric': 'auc',
	    'gamma':0,
	    'max_depth':4,
	    'lambda':700,
	    'subsample':0.7,
	    'colsample_bytree':0.3,
	    'min_child_weight':5,
	    'eta': 0.02,
	    'seed':1,
	    'nthread':8
	    }
	threads.append(Xgb_level_train_thread(config_instance,'level_two','log_move_xgb2000',X_0,X_1,uid_0,uid_1,params,2000))
	threads.append(Xgb_level_train_thread(config_instance,'level_two','log_move_xgb2500',X_0,X_1,uid_0,uid_1,params,2500))
	params={
	    'booster':'gbtree',
	    'objective': 'binary:logistic',
	   	'scale_pos_weight':13458.0/(1300.0),
	    'eval_metric': 'auc',
	    'gamma':0,
	    'max_depth':4,
	    'lambda':700,
	    'subsample':0.7,
	    'colsample_bytree':0.3,
	    'min_child_weight':5,
	    'eta': 0.02,
	    'seed':1,
	    'nthread':8
	    }
	threads.append(Xgb_level_train_thread(config_instance,'level_two','log_move_xgb2000_2',X_0,X_1,uid_0,uid_1,params,2000))
	threads.append(Xgb_level_train_thread(config_instance,'level_two','log_move_xgb2500_2',X_0,X_1,uid_0,uid_1,params,2500))
	for thread in threads:
		thread.run()

def level_three_train():
	config_instance=Config('log_move')
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
	threads.append(Level_train_thread(config_instance,LogisticRegression(solver='sag'),'level_three','log_move_lr_sag',X_0,X_1,uid_0,uid_1))

	for thread in threads:
		thread.run()
	pass
#预测

def level_one_predict():
	config_instance=Config('log_move')
	level='level_one'
	load_data_instance=load_data.Load_data(config_instance)
	X_0,X_1,uid_0,uid_1=load_data_instance.train_xy()
	predict_X,predict_uid=load_data_instance.predict_X()

	threads=[]
	threads.append(Level_predict_thread(config_instance,LogisticRegression(solver='sag'),level,'log_move_lr_sag',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,LogisticRegression(solver='newton-cg'),level,'log_move_lr_newton',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,LogisticRegression(solver='lbfgs'),level,'log_move_lr_lbfgs',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,LogisticRegression(solver='liblinear'),level,'log_move_lr_liblinear',X_0,X_1,predict_X,predict_uid))
	
	threads.append(Level_predict_thread(config_instance,RandomForestClassifier(n_estimators=100,max_depth=8,min_samples_split=9),level,'log_move_rf100',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,RandomForestClassifier(n_estimators=200,max_depth=8,min_samples_split=9),level,'log_move_rf200',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,RandomForestClassifier(n_estimators=500,max_depth=8,min_samples_split=9),level,'log_move_rf500',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,RandomForestClassifier(n_estimators=1000,max_depth=8,min_samples_split=9),level,'log_move_rf1000',X_0,X_1,predict_X,predict_uid))

	threads.append(Level_predict_thread(config_instance,GradientBoostingClassifier(n_estimators=20,max_depth=8,min_samples_split=9,learning_rate=0.02,subsample=0.7),level,'log_move_gbdt20',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,GradientBoostingClassifier(n_estimators=50,max_depth=8,min_samples_split=9,learning_rate=0.02,subsample=0.7),level,'log_move_gbdt50',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,GradientBoostingClassifier(n_estimators=100,max_depth=8,min_samples_split=9,learning_rate=0.02,subsample=0.7),level,'log_move_gbdt100',X_0,X_1,predict_X,predict_uid))

	threads.append(Level_predict_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=8,min_samples_split=9),n_estimators=20,learning_rate=0.02),level,'log_move_ada20',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=8,min_samples_split=9),n_estimators=50,learning_rate=0.02),level,'log_move_ada50',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=8,min_samples_split=9),n_estimators=100,learning_rate=0.02),level,'log_move_ada100',X_0,X_1,predict_X,predict_uid))

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
	threads.append(Xgb_level_predict_thread(config_instance,level,'log_move_xgb2000',X_0,X_1,predict_X,predict_uid,params,2000))
	threads.append(Xgb_level_predict_thread(config_instance,level,'log_move_xgb2500',X_0,X_1,predict_X,predict_uid,params,2500))
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
	threads.append(Xgb_level_predict_thread(config_instance,level,'log_move_xgb2000_2',X_0,X_1,predict_X,predict_uid,params,2000))
	threads.append(Xgb_level_predict_thread(config_instance,level,'log_move_xgb2500_2',X_0,X_1,predict_X,predict_uid,params,2500))

	for thread in threads:
		thread.run()

	pass

def level_two_predict():
	config_instance=Config('log_move')
	level='level_two'
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
	load_data_instance=load_train_data.Load_train_data(config_instance,'level_one',clf_name)
	X_0,X_1,uid_0,uid_1=load_data_instance.level_data()
	
	threads=[]
	threads.append(Level_predict_thread(config_instance,LogisticRegression(solver='sag'),level,'log_move_lr_sag',X_0,X_1,uid_0,uid_1))

	pass

def main():
	#level_one_wrapper()
	#level_two_wrapper()
	#level_three_train()
	level_one_predict()

if __name__ == '__main__':
	reload(sys)
	sys.setdefaultencoding('utf8')
	start=datetime.now()
	main()
	end=datetime.now()
	print "All Run time:"+str(float((end-start).seconds)/60.0)+"min / "+str(float((end-start).seconds))+"s"
