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

# import logging

# logging.basicConfig(level=logging.DEBUG,
#                 format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
#                 datefmt='%a, %d %b %Y %H:%M:%S',
#                 filename='predict_level_one.log',
#                 filemode='a+')

# console = logging.StreamHandler()
# console.setLevel(logging.INFO)
# formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# console.setFormatter(formatter)
# logging.getLogger('').addHandler(console)


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
		#logging.info('Begin train '+self.name)
		start=datetime.now()
		boost_instance=mboost.Mboost(self.config)
		boost_instance.xgb_level_train(self.level,self.name,self.X_0,self.X_1,self.uid_0,self.uid_1,self.params,self.round)
		end=datetime.now()
		#logging.info('End train '+self.name+", cost time: "+str(float((end-start).seconds)/60.0)+"min / "+str(float((end-start).seconds))+"s")