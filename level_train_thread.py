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
		#logging.info('Begin train '+self.name)
		start=datetime.now()
		boost_instance=mboost.Mboost(self.config)
		boost_instance.level_train(self.clf,self.level,self.name,self.X_0,self.X_1,self.uid_0,self.uid_1)
		end=datetime.now()
		#logging.info('End train '+self.name+", cost time: "+str(float((end-start).seconds)/60.0)+"min / "+str(float((end-start).seconds))+"s")