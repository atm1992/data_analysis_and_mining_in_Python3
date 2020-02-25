# -*- coding: UTF-8 -*-
import pandas as pd
# 三大分布
import scipy.stats as ss

def main():
	df = pd.read_csv("./data/HR.csv")
	# print df
	print('type(df):\n', type(df))
	print('\ntype(df["satisfaction_level"]):\n', type(df["satisfaction_level"]))
	print('\n均值df.mean():\n', df.mean())
	print('\ntype(df.mean()):\n', type(df.mean()))  # 返回的数据类型是Series
	print('\ndf["satisfaction_level"].mean():\n', df["satisfaction_level"].mean())
	print('\n中位数df.median():\n', df.median())
	print('\n中位数df["satisfaction_level"].median():\n', df["satisfaction_level"].median())
	print('\n上四分位数df.quantile(q=0.25):\n', df.quantile(q=0.25))
	print('\n上四分位数df["satisfaction_level"].quantile(q=0.25):\n', df["satisfaction_level"].quantile(q=0.25))
	print('\n众数df.mode():\n', df.mode()) # 众数可能会有多个，返回的行数取决于有最多个众数的那一列
	print('\n众数df["satisfaction_level"].mode():\n', df["satisfaction_level"].mode())
	print('\n众数df["department"].mode():\n', df["department"].mode())
	print('\n众数type(df["department"].mode()):\n', type(df["department"].mode()))

	# 离中趋势
	print('\n标准差df.std():\n', df.std())
	print('\n标准差df["satisfaction_level"].std():\n', df["satisfaction_level"].std())
	print('\n方差df.var():\n', df.var())
	print('\n方差df["satisfaction_level"].var():\n', df["satisfaction_level"].var())
	print('\n求和df.sum():\n', df.sum())  # 离散数据求和，是把每个值拼接起来
	print('\n求和df["satisfaction_level"].sum():\n', df["satisfaction_level"].sum())
	print('\n偏态系数df.skew():\n', df.skew())
	# 返回的值为负数，即负偏，均值小于中位数
	print('\n偏态系数df["satisfaction_level"].skew():\n', df["satisfaction_level"].skew())
	print('\n峰态系数df.kurt():\n', df.kurt())
	# 这里的峰态系数算法是将正态分布的峰态系数设为0
	# 返回的值为负数，即 该分布比正态分布更平缓些
	print('\n峰态系数df["satisfaction_level"].kurt():\n', df["satisfaction_level"].kurt())

	# 三大分布
	print('\n正态分布ss.norm:\n', ss.norm)  # 正态分布是个对象，有很多可调用的方法
	# mvsk：m表示均值，v表示方差，s表示偏态系数，k表示峰态系数
	print('\n正态分布的mvsk:\n', ss.norm.stats(moments="mvsk"))
	# pdf是指对于这个分布函数，指定一个横坐标的值，返回对应的纵坐标的值
	print('\n正态分布ss.norm.pdf(0):\n', ss.norm.pdf(0))
	# ppf的输入值必须是0到1之间，表示的是一个累计值。下例表示从负无穷大到所返回的值积分，结果为0.9
	print('\n正态分布ss.norm.ppf(0.9):\n', ss.norm.ppf(0.9))
	# cdf表示从负无穷一直积到2，累计概率为返回结果值
	print('\n正态分布ss.norm.cdf(2):\n', ss.norm.cdf(2))
	# 下例表示正两倍的标准差减去负两倍的标准差，这个范围内的累计概率
	print('\nss.norm.cdf(2) - ss.norm.cdf(-2):\n', ss.norm.cdf(2) - ss.norm.cdf(-2))
	# 得到10个符合正态分布的数字
	print('\nss.norm.rvs(size=10):\n', ss.norm.rvs(size=10))

	# 三大分布的操作和正态分布一样！ pdf ppf cdf rvs
	print('\n卡方分布ss.chi2:\n', ss.chi2)
	print('\nt分布ss.t:\n', ss.t)
	print('\nf分布ss.f:\n', ss.f)

	# 抽样
	print('\n抽样抽10个df.sample(10):\n', df.sample(10))
	# frac 为比例，总数乘以0.001 即为返回的个数
	print('\n抽样抽15个df.sample(frac=0.001):\n', df.sample(frac=0.001))
	print('\n抽样抽10个df["satisfaction_level"].sample(10):\n', df["satisfaction_level"].sample(10))


if __name__ == '__main__':
	main()
