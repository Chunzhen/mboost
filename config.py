#! /usr/bin/env python
# -*- coding:utf-8 -*-

#配置信息
import os

"""
class Config
训练信息配置类
"""
class Config(object):
	def __init__(self,scale):
		self.scale_level1=scale

		#路径
		self.path='F:/contest/rp/' #文件目录
		self.path_feature_type=self.path+'data/features_type.csv' #特征类型文件
		self.path_origin_train_x=self.path+'data/train_x.csv' #原始训练集特征文件
		self.path_train_x=self.path+'data/train_x_scale_'+self.scale_level1+'.csv' #变换后的训练特征文件
		self.path_train_y=self.path+'data/train_y.csv' #训练集类标签文件
		self.path_origin_predict_x=self.path+'data/test_x.csv' #原始测试集特征文件
		self.path_predict_x=self.path+'data/test_x_scale_'+self.scale_level1+'.csv'  #变换后的测试集特征文件
		self.path_uid=self.path+'data/uid.csv' #训练集uid

		#fold random_state
		self.fold_random_state=7#171 # n folds的随机种子
		self.n_folds=5 # n 划分数据集为多少折，本次实验统一划分为5折

		self.path_train=self.path+'train/' #训练输出文件夹
		self.path_predict=self.path+'predict/' #测试输出文件夹

		self.path_cor=self.path+'statistic/cor_log.csv' #高特征列与类标签相似度高于0.01的特征

	def init_path(self):
		"""
		初始化文件目录
		"""
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