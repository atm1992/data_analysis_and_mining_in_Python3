# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np


def main():
	df = pd.read_csv("./data/HR.csv")
	print('\ndf.head():\n', df.head())

	print("\nsatisfaction_level字段的分析!!!")
	sl_s = df["satisfaction_level"]
	print('\nsl_s.isnull():\n', sl_s.isnull())
	print('\nsl_s中值为NaN的行:\n', sl_s[sl_s.isnull()])
	print('\nsatisfaction_level的值为NaN的行:\n', df[df["satisfaction_level"].isnull()])
	sl_s = sl_s.dropna()  # 直接丢弃异常值
	# sl_s = sl_s.fillna(-1)  # 将异常值填充为指定值
	print('\nsl_s中值为NaN的行:\n', sl_s[sl_s.isnull()])
	print('\n均值sl_s.mean():\n', sl_s.mean())
	print('\n标准差sl_s.std():\n', sl_s.std())
	print('\n方差sl_s.var():\n', sl_s.var())
	print('\n最大值sl_s.max():\n', sl_s.max())
	print('\n最小值sl_s.min():\n', sl_s.min())
	print('\n中位数sl_s.median():\n', sl_s.median())
	print('\n下四分位数sl_s.quantile(q=0.25)):\n', sl_s.quantile(q=0.25))
	print('\n上四分位数sl_s.quantile(q=0.75)):\n', sl_s.quantile(q=0.75))
	print('\n偏度sl_s.skew():\n', sl_s.skew())  # 负偏
	print('\n峰度sl_s.kurt():\n', sl_s.kurt())  # 比正态分布平缓
	# bins=np.arange(0,1.1,0.1) 指定切分的临界值，从0到1.1，间隔为0.1
	# 直方图。返回结果的第二项为分割点，第一项为各分割区间的数据个数。最后一个区间为(0.9,1), 1.1是不包含的
	print('\n获取离散化分布:\n', np.histogram(sl_s.values, bins=np.arange(0, 1.1, 0.1)))

	print("\nlast_evaluation字段的分析!!!")
	le_s = df["last_evaluation"]
	print('\nle_s中值为NaN的行:\n', le_s[le_s.isnull()])  # 没有空值
	print('\n均值le_s.mean():\n', le_s.mean())
	print('\n标准差le_s.std():\n', le_s.std())
	print('\n方差le_s.var():\n', le_s.var())
	print('\n最大值le_s.max():\n', le_s.max())
	print('\n最小值le_s.min():\n', le_s.min())
	print('\n中位数le_s.median():\n', le_s.median())
	print('\n偏度le_s.skew():\n', le_s.skew())  # 极度正偏
	print('\n峰度le_s.kurt():\n', le_s.kurt())  # 数据非常不正常
	# 只返回了一个值， 可见该值是有很大问题的异常值。完全可以直接丢弃
	print('\nle_s中值大于1的行:\n', le_s[le_s > 1])
	# print(le_s[le_s<=1])
	le_q_low = le_s.quantile(q=0.25)  # 下四分位数
	le_q_high = le_s.quantile(q=0.75)  # 上四分位数
	le_q_iqr = le_q_high - le_q_low  # 四分位距
	k = 1.5  # 形变系数
	le_s = le_s[le_s < le_q_high + k * le_q_iqr][le_s > le_q_low - k * le_q_iqr]
	print('\nle_s中正常范围内的数据:\n', le_s)
	print('\nle_s中正常范围内的数据个数:\n', len(le_s))
	# 直方图。返回结果的第二项为分割点，第一项为各分割区间的数据个数。最后一个区间为(0.9,1), 1.1是不包含的
	print('\n获取离散化分布:\n', np.histogram(le_s.values, bins=np.arange(0, 1.1, 0.1)))
	# 去掉异常值之后，再看一下数据的均值、标准差、中位数之类的

	print("\nnumber_project字段的分析!!!")
	np_s = df["number_project"]
	print('\nnp_s中值为NaN的行:\n', np_s[np_s.isnull()])  # 没有空值
	print('\n均值np_s.mean():\n', np_s.mean())
	print('\n标准差np_s.std():\n', np_s.std())
	print('\n方差np_s.var():\n', np_s.var())
	print('\n最大值np_s.max():\n', np_s.max())
	print('\n最小值np_s.min():\n', np_s.min())
	print('\n中位数np_s.median():\n', np_s.median())
	print('\n偏度np_s.skew():\n', np_s.skew())  # 稍微正偏
	print('\n峰度np_s.kurt():\n', np_s.kurt())
	# 因为属性值都是离散的整数
	print('\nnp_s中每个数字分别出现了多少次:\n', np_s.value_counts())
	# 静态结构分析。normalize=True 标准化、归一化
	print('\nnp_s中每个数字的各自占比:\n', np_s.value_counts(normalize=True))
	print('\nnp_s中每个数字的各自占比(按索引排序):\n', np_s.value_counts(normalize=True).sort_index())

	print("\naverage_monthly_hours字段的分析!!!")
	amh_s = df["average_monthly_hours"]
	print('\namh_s中值为NaN的行:\n', amh_s[amh_s.isnull()])  # 没有空值
	print('\n均值amh_s.mean():\n', amh_s.mean())
	print('\n标准差amh_s.std():\n', amh_s.std())
	print('\n方差amh_s.var():\n', amh_s.var())
	print('\n最大值amh_s.max():\n', amh_s.max())
	print('\n最小值amh_s.min():\n', amh_s.min())
	print('\n中位数amh_s.median():\n', amh_s.median())
	print('\n偏度amh_s.skew():\n', amh_s.skew())  # 稍微正偏
	print('\n峰度amh_s.kurt():\n', amh_s.kurt())  # 比较平缓
	amh_q_low = amh_s.quantile(q=0.25)
	amh_q_high = amh_s.quantile(q=0.75)
	amh_q_iqr = amh_q_high - amh_q_low
	k = 1.5
	# 剔除异常值
	amh_s = amh_s[amh_s < amh_q_high + k * amh_q_iqr][amh_s > amh_q_low - k * amh_q_iqr]
	print('\namh_s中正常值的个数:\n', len(amh_s))
	# 直方图。返回结果的第二项为分割点，第一项为各分割区间的数据个数
	# bins=10 表示分成10份
	print('\n获取离散化分布:\n', np.histogram(amh_s.values, bins=10))
	# 使用value_counts实现和上面一样的功能。bins的取值方式不一样
	print('\namh_s中数据的离散化分布:\n', amh_s.value_counts(bins=10))

	print("\ntime_spend_company字段的分析!!!")
	tsc_s = df["time_spend_company"]
	print('\ntsc_s中每个数字分别出现了多少次:\n', tsc_s.value_counts().sort_index())
	print('\ntsc_s中值为NaN的行:\n', tsc_s[tsc_s.isnull()])  # 没有空值
	print('\n均值tsc_s.mean():\n', tsc_s.mean())
	print('\n标准差tsc_s.std():\n', tsc_s.std())
	print('\n方差tsc_s.var():\n', tsc_s.var())
	print('\n最大值tsc_s.max():\n', tsc_s.max())
	print('\n最小值tsc_s.min():\n', tsc_s.min())
	print('\n中位数tsc_s.median():\n', tsc_s.median())
	print('\n偏度tsc_s.skew():\n', tsc_s.skew())  # 稍微正偏
	print('\n峰度tsc_s.kurt():\n', tsc_s.kurt())

	print("\nWork_accident字段的分析!!!")
	wa_s = df["Work_accident"]
	print('\nwa_s中每个数字分别出现了多少次:\n', wa_s.value_counts().sort_index())
	print('\nwa_s中值为NaN的行:\n', wa_s[wa_s.isnull()])  # 没有空值
	print('\n均值wa_s.mean():\n', wa_s.mean())  # 工作事故率
	print('\n标准差wa_s.std():\n', wa_s.std())
	print('\n方差wa_s.var():\n', wa_s.var())
	print('\n最大值wa_s.max():\n', wa_s.max())
	print('\n最小值wa_s.min():\n', wa_s.min())
	print('\n中位数wa_s.median():\n', wa_s.median())
	print('\n偏度wa_s.skew():\n', wa_s.skew())  # 稍微正偏
	print('\n峰度wa_s.kurt():\n', wa_s.kurt())

	print("\nleft字段的分析!!!")
	l_s = df["left"]
	print('\nl_s中每个数字分别出现了多少次:\n', l_s.value_counts().sort_index())
	print('\nl_s中值为NaN的行:\n', l_s[l_s.isnull()])  # 没有空值
	print('\n均值l_s.mean():\n', l_s.mean())
	print('\n标准差l_s.std():\n', l_s.std())
	print('\n方差l_s.var():\n', l_s.var())
	print('\n最大值l_s.max():\n', l_s.max())
	print('\n最小值l_s.min():\n', l_s.min())
	print('\n中位数l_s.median():\n', l_s.median())
	print('\n偏度l_s.skew():\n', l_s.skew())  # 稍微正偏
	print('\n峰度l_s.kurt():\n', l_s.kurt())

	print("\npromotion_last_5years字段的分析!!!")  # 最近5年是否升职
	pl5_s = df["promotion_last_5years"]
	# 查看数据的分布
	print('\npl5_s中每个数字分别出现了多少次:\n', pl5_s.value_counts().sort_index())

	print("\nsalary字段的分析!!!")
	s_s = df["salary"]
	# 查看数据的分布。有一个异常值nme
	print('\ns_s中每个数字分别出现了多少次:\n', s_s.value_counts())
	s_s.where(s_s == "nme")  # 符合条件的显示真实值，不符合条件的显示为NaN
	print('\n将s_s中的异常值设置为NaN（空值）:\n', s_s.where(s_s != "nme"))
	s_s = s_s.where(s_s != "nme").dropna()  # 删除异常值
	print('\n删除s_s中的异常值后:\n', s_s)
	print('\ns_s中每个数字分别出现了多少次:\n', s_s.value_counts())

	print("\ndepartment字段的分析!!!")
	d_s = df["department"]
	# 查看数据的分布。有一个值为sale，还有一个为sales
	print('\nd_s中每个数字的各自占比:\n', d_s.value_counts(normalize=True))
	# 剔除异常值sale
	d_s = d_s.where(d_s != "sale").dropna()
	print('\n删除d_s中的异常值后:\n', d_s)


if __name__ == '__main__':
	main()
