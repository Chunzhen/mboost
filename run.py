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
import mboost

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression


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

class Level_train_thread(threading.Thread):
	def __init__(self,config,clf,level,name,X_0,X_1,uid_0,uid_1):
		threading.Thread.__init__(self)
		self.config=config
		self.clf=clf
		self.level=level
		self.name=name
		self.X_0=X_0
		self.X_1=X_1
		self.uid_0=uid_0
		self.uid_1=uid_1

	def run(self):
		boost_instance=mboost.Mboost(self.config)
		boost_instance.level_train(self.clf,self.level,self.name,self.X_0,self.X_1,self.uid_0,self.uid_1)

#训练
def level_one_wrapper():
	config_instance=Config('log')
	load_data_instance=load_data.Load_data(config_instance)
	X_0,X_1,uid_0,uid_1=load_data_instance.train_xy()
	
	threads=[]
	thread1=Level_train_thread(config_instance,LogisticRegression(solver='sag'),'level_one','lr_sag',X_0,X_1,uid_0,uid_1) #lr最好的solver
	threads.append(thread1)

	for thread in threads:
		thread.start()

	for thread in threads:
		thread.join()

#预测

def main():
	level_one_wrapper()

if __name__ == '__main__':
	reload(sys)
	sys.setdefaultencoding('utf8')
	start=datetime.now()
	main()
	end=datetime.now()
	print "All Run time:"+str(float((end-start).seconds)/60.0)+"min / "+str(float((end-start).seconds))+"s"
