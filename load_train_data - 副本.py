#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import time

import load_data
from config import Config

from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from pylab import *  
mpl.rcParams['font.sans-serif'] = ['SimHei'] 
  
mpl.rcParams['axes.unicode_minus'] = False

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

	def level_ranks(self,column_dict):
		column_dict2=sorted(column_dict.items(),key=lambda d:d[1])
		i=0
		ranks={}
		for uid, score in column_dict2:
			ranks[uid]=i
			i+=1

		return ranks

	def level_data_part(self):
		"""
		读取上一层多个训练器的输出结果，作为下一层的训练特征
		"""
		level=self.level
		clf_name=self.__clf_name
		load_data_instance=load_data.Load_data(self.config)
		y,uids=load_data_instance.train_y()
		X_00,X_11,uid_00,uid_11=load_data_instance.train_xy()
		X,uid__=load_data_instance.train_X()

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
		diff_uid=set([])
		for name in clf_name:
			column_dict=self.load_clf_file(level,name) #预测dict: uid->score
			column_score=self.load_clf_score(level,name) #预测n folds auc
			column_important.append(column_score)

			column_rank=self.level_ranks(column_dict)
			lr_dict=self.load_clf_file(level,'log_move_lr_sag')
			lr_rank=self.level_ranks(lr_dict)

			lr_dict2=self.load_clf_file('level_two','log_move_lr_sag')
			lr_rank2=self.level_ranks(lr_dict2)
			#print lr_rank

			print name,"  ",column_score
			column_dict2=sorted(column_dict.items(),key=lambda d:d[1])
			#print column_dict2
			i=0
			one_diff=[]
			zero_diff=[]
			one_index=[]
			zero_index=[]
			yy=[]
			scores=[]
			bingo_rank=[]

			bingo_y=[]
			bingo_scores=[]

			for uid,score in column_dict2:
				temp=d.get(uid,[])
				temp.append(column_dict[uid])
				d[uid]=temp

				diff=column_rank[uid]-lr_rank[uid]
				if uid in uid_00:
					#print diff,' ','y=',0
					zero_diff.append(diff)
					zero_index.append(i)
					yy.append(0)
				else:
					#print diff,' ','y=',1
					one_diff.append(diff)
					one_index.append(i)
					yy.append(1)
				#print diff
				if diff>2500+i*0.15:  #lr diff>2500+i*0.2
					diff_uid.add(uid)

					if uid in uid_00:
						bingo_y.append(0)
						pass
					else:
						bingo_y.append(1)
						pass
					
						#print lr_rank2[uid],' ',column_rank[uid]
						#bingo_rank.append(lr_rank2[uid])
					bingo_scores.append(score)
					if lr_rank2[uid]<200: #or lr_rank2[uid]>2282
						#print 'bingo'
						#print lr_rank2[uid],' ',column_rank[uid]
						score=-100

				scores.append(score)

				i+=1

			idex=0
			auc_score=metrics.roc_auc_score(yy,scores)
			#auc_bingo=metrics.roc_auc_score(bingo_y,bingo_scores)
			#print "auc:",auc_score,' auc_bingo:',auc_bingo
			print sorted(bingo_rank)
			self.print_diff(zero_diff[idex:],zero_index[idex:],one_diff[idex:],one_index[idex:])
		X_0=[]
		X_1=[]
		uid_0=[]
		uid_1=[]
		print len(diff_uid)

		# #将类0与类1拆分到不同数组
		for i in range(len(y)):
			if uids[i] in diff_uid:
				if y[i]==0:
					#print i
					X_1.append(X[i])
					uid_1.append(uids[i])
				else:
					X_0.append(X[i])
					uid_0.append(uids[i])

		return np.array(X_0),np.array(X_1),np.array(uid_0),np.array(uid_1)

	def print_diff(self,zero_diff,zero_index,one_diff,one_index):
		"""
		:type zero_diff: List[int] 0类的排名分布差
		:type zero_index: List[int] 0类的下标
		:type one_diff: List[int] 1类的排名分布差
		:type one_index: List[int] 1类的下标
		"""
		x=[]
		y=[]
		for i in range(3000,15000):
			x.append(i)
			y.append(2500+i*0.2)

		plt.plot(x,y,color='yellow',linewidth=3)
		plt.title(u'XGBoost 2000与LR的排名差')
		plt.xlabel(u'rank')
		plt.ylabel(u'rank diff')

		plt.scatter(zero_index,zero_diff,label='正样本')
		plt.scatter(one_index,one_diff,c='red',label='负样本')
		plt.legend(loc='upper center')
		plt.show()

def  main():
	"""
	本地测试函数
	"""
	reload(sys)
	sys.setdefaultencoding('utf8')
	ftype='log_move'
	config_instance=Config(ftype)
	level='level_one'
	clf_name=[
		#ftype+'_lr_sag',
		# ftype+'_lr_newton',
		# ftype+'_lr_lbfgs',
		# ftype+'_lr_liblinear',
		# ftype+'_rf100',
		# ftype+'_rf200',
		# ftype+'_rf500',
		# ftype+'_rf1000',
		# ftype+'_gbdt20',
		# ftype+'_gbdt50',
		# ftype+'_gbdt100',
		# ftype+'_ada20',
		# ftype+'_ada50',
		# ftype+'_ada100',
		ftype+'_xgb2000',
		# ftype+'_xgb2500',
		# ftype+'_xgb2000_2',
		# ftype+'_xgb2500_2'
	]

	load_data_instance=Load_train_data(config_instance,level,clf_name)
	X_0,X_1,uid_0,uid_1=load_data_instance.level_data_part()
	pass

if __name__ == '__main__':
	main()
	

