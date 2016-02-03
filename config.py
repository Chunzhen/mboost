#! /usr/bin/env python
# -*- coding:utf-8 -*-

#配置信息

#scale
class Config(object):
	def __init__(self,scale):
		self.scale_level1=scale

		#路径
		self.path='F:/contest/rp/'
		self.path_feature_type=self.path+'data/features_type.csv'
		self.path_origin_train_x=self.path+'data/train_x.csv'
		self.path_train_x=self.path+'data/train_x_scale_'+self.scale_level1+'.csv'
		self.path_train_y=self.path+'data/train_y.csv'
		self.path_origin_predict_x=self.path+'data/test_x.csv'
		self.path_predict_x=self.path+'data/test_x_scale_'+self.scale_level1+'.csv'
		self.path_uid=self.path+'data/uid.csv'

