#-*- coding: UTF-8 -*-
import numpy as np
import scipy.stats as ss
from statsmodels.graphics.api import qqplot
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from scipy import linalg    # 用于线性计算的包


# 正态性检验
def normal_test():
    # 符合标准正态分布的20个数
    norm_dist = ss.norm.rvs(size=20)
    print(norm_dist)
    # 检测是否符合正态分布。查看检验统计量statistic、p值pvalue
    # pvalue 大于显著性水平 0.05，因此符合正态分布假设
    # 这里的normaltest使用的并不是u检验法，而是基于偏度、峰度的检验
    print('\n正态检验结果：\n', ss.normaltest(norm_dist))


# 卡方检验
def chi2_test():
    # 可得到检验统计量、p值、自由度、理论分布
    print('\n卡方检验结果：\n', ss.chi2_contingency([[15, 95], [85, 5]]))


# 两独立样本t检验。检验两组值的均值是否存在较大差异
def t_test():
    # 返回检验统计量、p值
    print('\nt检验结果：\n', ss.ttest_ind(ss.norm.rvs(size=100), ss.norm.rvs(size=200)))


# 方差分析（F检验）。
def f_test():
    # 返回检验统计量、p值
    print('\n方差F检验结果：\n', ss.f_oneway([49, 50, 39, 40, 43], [28, 32, 30, 26, 34], [38, 40, 45, 42, 48]))


# QQ图。默认检验一个分布是否符合正态分布
def qq_plot():
    plt.show(qqplot(ss.norm.rvs(size=100)))


# 相关系数
def correlation_coefficient():
    # Series.corr(self, other, method='pearson', min_periods=None)
    s1 = pd.Series([0.1, 0.2, 1.1, 2.4, 1.3, 0.3, 0.5])
    s2 = pd.Series([0.5, 0.4, 1.2, 2.5, 1.1, 0.7, 0.1])
    # 默认为皮尔逊相关系数
    print('\ns1和s2之间的皮尔逊相关系数：\n', s1.corr(s2))
    print('\ns1和s2之间的斯皮尔曼相关系数：\n', s1.corr(s2, method="spearman"))

    # DataFrame.corr(self, method='pearson', min_periods=1)
    df = pd.DataFrame([s1, s2])
    # DataFrame是针对列进行相关性计算的
    print('\nDataFrame默认针对列的相关系数：\n', df.corr())
    # 针对上面的改进方法。用np.array先进行转置，再构造一个DataFrame
    df = pd.DataFrame(np.array([s1, s2]).T)
    print(df)
    print('\nDataFrame转置后针对行的相关系数：\n', df.corr())
    print('\nDataFrame转置后针对行的斯皮尔曼相关系数：\n', df.corr(method="spearman"))


# 线性回归
def regression():
    # 0~9的10个数，转换成float类型，把每个元素当作一个数组来看
    x = np.arange(10).astype(np.float).reshape((10, 1))
    # 3 和 4 是要求解的系数，再加上一些随机的噪声(0~1)
    y = x*3 + 4 + np.random.random((10, 1))

    reg = LinearRegression().fit(x, y)
    y_pred = reg.predict(x)    # 预测值
    print('\n实际值：\n', y)
    print('\n预测值：\n', y_pred)
    print(dir(reg))
    print('\n参数：\n', reg.coef_)
    print('\n截距：\n', reg.intercept_)


# PCA降维
def pca():
    # 要记得进行转置
    data = np.array([np.array([2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2, 1, 1.5, 1.1]), np.array([2.4, 0.7, 2.9, 2.2, 3, 2.7, 1.6, 1.1, 1.6, 0.9])]).T
    print('\n输入数据data：\n', data)
    # 降成1维
    lower_dim = PCA(n_components=1)
    lower_dim.fit(data)
    # 降维后，保留了96.3%的信息量
    print('\n维度的重要性：\n', lower_dim.explained_variance_ratio_)
    print('\n转换后的数值：\n', lower_dim.fit_transform(data))


# sklearn里的PCA算法用到的是奇异值分解（SVD）的方法，而并不是PCA方法
# 自定义的PCA方法。指定一个很大的维数，如果没有这么多维度，就全取
def my_pca(data, n_components=1000000):
    # 得到data的均值，axis=0 表示每个属性（维度）的均值，即 针对列取均值
    mean_vals = np.mean(data, axis=0)
    # 中心化，每一维度上的所有数据都减去该维度的均值
    mid = data - mean_vals
    # 求协方差。需要指定rowvar=False，表示针对列进行协方差计算。若不指定，则针对行进行协方差计算
    cov_mat = np.cov(mid, rowvar=False)
    # 求协方差矩阵的特征值（一维数组）和特征向量（二维数组）。把cov_mat转换成矩阵
    eig_vals, eig_vects = linalg.eig(np.mat(cov_mat))
    # 使用argsort得到按值升序后的索引（下标），而使用sort是得到排序后的值
    eig_val_index = np.argsort(eig_vals)
    # 取出最大的一个特征值所在的索引（下标）
    eig_val_index = eig_val_index[-1:-(n_components + 1):-1]
    # 取出最大的特征值所对应的特征向量。第eig_val_index列的所有行
    eig_vects = eig_vects[:, eig_val_index]
    # 进行转换，中心化后的矩阵投影到最大的特征值所对应的特征向量上
    low_dim_mat = np.dot(mid, eig_vects)
    # 返回降维后的（一维）矩阵 以及 对应的特征值
    return low_dim_mat, eig_vals[eig_val_index]


def main():
    data = np.array([np.array([2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2, 1, 1.5, 1.1]), np.array([2.4, 0.7, 2.9, 2.2, 3, 2.7, 1.6, 1.1, 1.6, 0.9])]).T
    # 测试一下自定义的PCA方法my_pca。最终保留一维
    print('\n测试自定义的PCA方法my_pca：\n', my_pca(data, n_components=1))


if __name__ == '__main__':
    # normal_test()    # 正态性检验
    # chi2_test()   # 卡方检验
    # t_test()    # 独立t分布检验
    # f_test()   # 方差检验（也叫 F检验）
    # qq_plot()    # QQ图
    # correlation_coefficient()  # 相关系数
    # regression()   # 线性回归
    # pca()   # PCA降维
    main()    # 测试一下自定义的PCA方法my_pca
