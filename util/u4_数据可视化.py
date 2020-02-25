# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 柱状图
def bar_chart():
	df = pd.read_csv("./data/HR.csv")
	# 删除空值以及异常值
	df = df.dropna(axis=0, how="any")
	df = df[df["last_evaluation"] <= 1][df["salary"] != "nme"][df["department"] != "sale"]

	# 用seaborn来改变样式。style 可以为 whitegrid、darkgrid、dark、white、ticks
	sns.set_style(style="darkgrid")
	# 设置字体、字号。context 可以为 paper、notebook、talk、poster
	sns.set_context(context="poster", font_scale=0.8)
	# 设置调色板。填入的参数是个数组
	# https://matplotlib.org/tutorials/colors/colormaps.html#sphx-glr-tutorials-colors-colormaps-py
	# http://seaborn.pydata.org/generated/seaborn.color_palette.html#seaborn.color_palette
	sns.set_palette([sns.color_palette("RdBu", n_colors=7)[5]])
	# sns.countplot(x="salary", data=df)
	# 多层绘制。以部门为分割进行count
	sns.countplot(x="salary", hue="department", data=df)
	plt.show()

	# 以下是用matplotlib来绘图
	# plt.title("SALARY")
	# plt.xlabel("salary")
	# plt.ylabel("count")
	# # 在X轴上进行标注，第一项为标注位置，第二项为标注的内容
	# plt.xticks(np.arange(len(df["salary"].value_counts()))+0.5,df["salary"].value_counts().index)
	# # 设置显示范围。X轴的最小值、最大值，Y轴的最小值、最大值
	# plt.axis([0,4,0,10000])
	# # 定义柱状图（条形图）的每个柱的位置、高度，以及每个柱的宽度
	# plt.bar(np.arange(len(df["salary"].value_counts()))+0.5,df["salary"].value_counts(),width=0.5)
	# # 把数字标注在图中每个柱的上方。标注的位置和标注的值
	# for x,y in zip(np.arange(len(df["salary"].value_counts()))+0.5,df["salary"].value_counts()):
	# 	plt.text(x,y,y,ha="center",va="bottom")
	# plt.show()

# 直方图
def histogram():
	df = pd.read_csv("./data/HR.csv")
	# 删除空值以及异常值
	df = df.dropna(axis=0, how="any")
	df = df[df["last_evaluation"] <= 1][df["salary"] != "nme"][df["department"] != "sale"]

	sns.set_style(style="whitegrid")
	sns.set_context(context="poster", font_scale=0.8)
	sns.set_palette(sns.color_palette("RdBu", n_colors=7))

	f = plt.figure()
	f.add_subplot(1,3,1)
	# 分布曲线 以及 直方图。kde默认为True，显示分布曲线；为False时，不显示分布曲线
	# hist默认为True，显示直方图；为False时，不显示直方图
	sns.distplot(df["satisfaction_level"],bins=10)
	f.add_subplot(1,3,2)
	sns.distplot(df["last_evaluation"],bins=10)
	f.add_subplot(1, 3, 3)   # 一行三列，第三个
	sns.distplot(df["average_monthly_hours"], bins=10)
	plt.show()

# 箱线图
def box_plot():
	df = pd.read_csv("./data/HR.csv")
	# 删除空值以及异常值
	df = df.dropna(axis=0, how="any")
	df = df[df["last_evaluation"] <= 1][df["salary"] != "nme"][df["department"] != "sale"]

	sns.set_style(style="whitegrid")
	sns.set_context(context="poster", font_scale=0.8)
	sns.set_palette(sns.color_palette("RdBu", n_colors=7))

	# sns.boxplot(y=df["time_spend_company"])
	sns.boxplot(x=df["time_spend_company"],saturation=0.75,whis=3)
	plt.show()

# 折线图  matplotlib 中的 plot()
def line_chart():
	df = pd.read_csv("./data/HR.csv")
	# 删除空值以及异常值
	df = df.dropna(axis=0, how="any")
	df = df[df["last_evaluation"] <= 1][df["salary"] != "nme"][df["department"] != "sale"]

	sns.set_style(style="whitegrid")
	sns.set_context(context="poster", font_scale=0.8)
	sns.set_palette(sns.color_palette("RdBu", n_colors=7))

	# sub_df = df.groupby("time_spend_company").mean()
	# # 横轴，纵轴
	# sns.pointplot(sub_df.index,sub_df["left"])

	# 另一种画法。一定要指定data
	sns.pointplot(x="time_spend_company",y="left",data=df)
	plt.show()

# seaborn 中没有饼图的画法，因此使用matplotlib来绘制
# 饼图。主要用来做结构分析，主要是一些类别
def pie_chart():
	df = pd.read_csv("./data/HR.csv")
	# 删除空值以及异常值
	df = df.dropna(axis=0, how="any")
	df = df[df["last_evaluation"] <= 1][df["salary"] != "nme"][df["department"] != "sale"]

	sns.set_style(style="whitegrid")
	sns.set_context(context="poster", font_scale=0.8)
	sns.set_palette(sns.color_palette("RdBu", n_colors=7))

	lbs = df["salary"].value_counts().index
	# 设定离开的间隔，起着重强调的作用
	explodes = [0.1 if i=="low" else 0 for i in lbs]
	# labels 显示label，autopct 显示数字（设定格式），colors指定色系
	plt.pie(df["salary"].value_counts(normalize=True),explode=explodes,labels=lbs,autopct="%1.1f%%",colors=sns.color_palette("Reds"))
	plt.show()

if __name__ == '__main__':
	# bar_chart()   # 柱状图
	# histogram()   # 直方图
	# box_plot()    # 箱线图
	# line_chart()   # 折线图
	pie_chart()    # 饼图
