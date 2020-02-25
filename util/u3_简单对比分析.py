# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd


def main():
	# 在对比分析之前，需要先把异常值都删除掉
	df = pd.read_csv("./data/HR.csv")
	print('\ndf.head():\n', df.head())

	# axis=0 表示以行为单位进行删除，即删除空值所在行
	# axis=1 表示以列为单位进行删除
	# how = "all" 表示该行都为空值时，才删除该行
	# how="any" 表示只要该行存在空值，就删除该行
	df = df.dropna(axis=0, how="any")
	print('\n删除df中的空值后:\n', df)
	df = df[df["last_evaluation"] <= 1][df["salary"] != "nme"][df["department"] != "sale"]
	print('\n删除df中的异常值后:\n', df)

	# 下面开始简单的对比分析
	# 以部门为单位进行对比
	# 对部门进行分组，分组之后要使用聚合函数。跟SQL一样
	print('\n对部门分组后的均值（文字的、离散的、非数值的属性被自动去掉了）:\n', df.groupby("department").mean())

	# 进行切片，只显示指定的属性。:, 表示index全选
	print('\n只显示指定属性的分组信息:\n', df.loc[:,["last_evaluation","department"]].groupby("department").mean())
	# 自定义函数进行对比。比如这里的极差
	df_amh = df.loc[:,["average_monthly_hours","department"]].groupby("department")["average_monthly_hours"].apply(lambda x:x.max()-x.min())
	print('\n自定义函数进行对比（此处自定义了极差）:\n', df_amh)

if __name__ == '__main__':
	main()
