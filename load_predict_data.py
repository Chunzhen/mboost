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

class Load_predict_data(object):
	"""
	:class Load_predict_data
	:读取第2-n层的预测数据集的前一层预测结果作为下一层模型的特征
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
		reader=pd.read_csv(self.config.path_predict+level+'/'+name+'.csv',iterator=False,delimiter=',',encoding='utf-8')
		d={}
		for i in range(len(reader['uid'])):
			d[reader['uid'][i]]=np.log10(reader['score'][i])
			#d[reader['uid'][i]]=reader['score'][i]
		return d

	def level_data(self):
		"""
		读取上一层多个训练器的输出结果，作为下一层的训练特征
		"""
		level=self.level
		clf_name=self.__clf_name
		load_data_instance=load_data.Load_data(self.config)
		predict_X,uids=load_data_instance.predict_X()
		
		"""
		注释代码为划分本地验证集后的数据读取，先将数据分离出20%作为本地的验证集
		但线上预测时，为更多使用数据，并没有本地验证集
		如果本地训练，则预测集为本地的验证集
		"""
		# X_0,test_X_0,X_1,test_X_1,uid_0,test_uid_0,uid_1,test_uid_1=load_data_instance.train_test_xy(1)
		# predict_X=np.vstack((test_X_0,test_X_1))
		# predict_uid=np.hstack((test_uid_0,test_uid_1))
		# uids=predict_uid.astype('int')

		d={}
		for name in clf_name:
			column_dict=self.load_clf_file(level,name)
			for uid in uids:
				temp=d.get(uid,[])
				temp.append(column_dict[uid])
				d[uid]=temp
		
		X=[]
		for i in range(len(uids)):
			X.append(d[uids[i]])

		return np.array(X),np.array(uids)

def main():
	"""
	本地测试函数
	"""
	config_instance=Config('log_move')
	level='level_one'
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
	predict_data_instance=Load_predict_data(config_instance,'level_one',clf_name)
	predict_X,predict_uid=predict_data_instance.level_data()
	print predict_X
	pass

if __name__ == '__main__':
	main()