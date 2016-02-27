#! /usr/bin/env python
# -*- coding:utf-8 -*-

#配置信息
import os

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

		#fold random_state
		self.fold_random_state=7#171
		self.n_folds=5

		self.path_train=self.path+'train_local/'
		self.path_predict=self.path+'predict/'

		self.path_cor=self.path+'statistic/cor_log.csv'

	def init_path(self):
		paths=[self.path_train,self.path_predict,self.path_train+'level_one/',self.path_train+'level_two/']
		for path in paths:
			if not os.path.exists(path):
				os.mkdir(path)

def main():
	instance=Config('log')
	instance.init_path()
	pass

if __name__ == '__main__':
	main()