#! /usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import time

import load_data
from config import Config

from sklearn.cross_validation import train_test_split

class Load_train_data(object):
	"""
	:class Load_train_data
	:读取第2-n层数据类，将前一层的输出结果作为特征读取
	"""
	def __init__(self,config,level,clf_name):
		"""
		:type config: Config 配置信息
		:type level: str 读取第几层的数据
		:type clf_name: List[str] 前一层的分类器（或回归器）的命名集合
		"""
		self.config=config
		self.level=level
		self.__clf_name=clf_name

	def load_clf_file(self,level,name):
		"""
		:type level: str 读取第几层的数据
		:type name: str 分类器命名
		:读取上一层一个模型的预测结果作为一维特征，作log变换对下一层模型的描述更加稳定
		"""
		reader=pd.read_csv(self.config.path_train+level+'/'+name+'.csv',iterator=False,delimiter=',',encoding='utf-8',header=None)
		d={}
		for i in range(len(reader[0])):
			d[str(reader[0][i])]=np.log10(reader[1][i])
			#d[str(reader[0][i])]=reader[1][i]
		return d

	def load_clf_score(self,level,name):
		"""
		:type level: str 读取第几层的数据
		:type name: str 分类器命名
		:读取上一层一个模型n folds训练，folds的预测AUC
		"""
		reader=pd.read_csv(self.config.path_train+level+'/'+name+'_score.csv',iterator=False,delimiter=',',encoding='utf-8',header=None)
		return np.mean(reader[0])

	def level_data(self):
		"""
		读取上一层多个训练器的输出结果，作为下一层的训练特征
		"""
		level=self.level
		clf_name=self.__clf_name
		load_data_instance=load_data.Load_data(self.config)
		y,uids=load_data_instance.train_y()
		X_00,X_11,uid_00,uid_11=load_data_instance.train_xy()

		"""
		注释代码为划分本地验证集后的数据读取，先将数据分离出20%作为本地的验证集
		但线上预测时，为更多使用数据，并没有本地验证集
		"""
		# X_00,test_X_00,X_11,test_X_11,uid_00,test_uid_00,uid_11,test_uid_11=load_data_instance.train_test_xy(1)
		# uids=np.hstack((uid_00,uid_11))
		# print len(uids)+len(test_uid_00)+len(test_uid_11)
		# y=np.hstack((np.ones(len(X_00)),np.zeros(len(X_11))))

		column_important=[]
		d={}
		for name in clf_name:
			column_dict=self.load_clf_file(level,name) #预测dict: uid->score
			column_score=self.load_clf_score(level,name) #预测n folds auc
			column_important.append(column_score)
			print name,"  ",column_score

			for uid in uids:
				temp=d.get(uid,[])
				temp.append(column_dict[uid])
				d[uid]=temp

		X_0=[]
		X_1=[]
		uid_0=[]
		uid_1=[]

		#将类0与类1拆分到不同数组
		for i in range(len(y)):
			if y[i]==0:
				X_1.append(d[uids[i]])
				uid_1.append(uids[i])
			else:
				X_0.append(d[uids[i]])
				uid_0.append(uids[i])

		return np.array(X_0),np.array(X_1),np.array(uid_0),np.array(uid_1)

def  main():
	"""
	本地测试函数
	"""
	ftype='log_move'
	config_instance=Config(ftype)
	level='level_one'
	clf_name=[
		ftype+'_lr_sag',
		ftype+'_lr_newton',
		ftype+'_lr_lbfgs',
		ftype+'_lr_liblinear',
		ftype+'_rf100',
		ftype+'_rf200',
		ftype+'_rf500',
		ftype+'_rf1000',
		ftype+'_gbdt20',
		ftype+'_gbdt50',
		ftype+'_gbdt100',
		ftype+'_ada20',
		ftype+'_ada50',
		ftype+'_ada100',
		ftype+'_xgb2000',
		ftype+'_xgb2500',
		ftype+'_xgb2000_2',
		ftype+'_xgb2500_2'
	]

	load_data_instance=Load_train_data(config_instance,level,clf_name)
	X_0,X_1,uid_0,uid_1=load_data_instance.level_data()
	pass

if __name__ == '__main__':
	main()
	

