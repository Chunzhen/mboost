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


class Xgb_level_train_thread(object):
	"""
	:class Xgb_level_train_thread
	:层次训练的多线程类
	:由于本人机器不太好，所以改为单线程运行
	:只要将注释代码去掉注释，类继承object改为threading.Thread即可改为多线程运行
	"""
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
		#logging.info('Begin train '+self.name)
		start=datetime.now()
		boost_instance=mboost.Mboost(self.config)
		boost_instance.xgb_level_train(self.level,self.name,self.X_0,self.X_1,self.uid_0,self.uid_1,self.params,self.round)
		end=datetime.now()
		#logging.info('End train '+self.name+", cost time: "+str(float((end-start).seconds)/60.0)+"min / "+str(float((end-start).seconds))+"s")