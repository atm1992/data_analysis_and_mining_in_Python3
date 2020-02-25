#-*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context(font_scale=1.5)
df = pd.read_csv("./data/HR.csv")
"""
交叉分析（分析属性与属性之间的关系）

# 使用 t 检验来分析不同部门之间的离职率是否有关
# 按部门分组之后的索引
df_indices = df.groupby(by='department').indices
# 销售部的离职率
sales_values = df['left'].iloc[df_indices['sales']].values
# 技术部的离职率
technical_values = df['left'].iloc[df_indices['technical']].values
# 销售部与技术部之间的p值
print(ss.ttest_ind(sales_values,technical_values)[1])
# 所有部门的两两之间求p值
df_keys = list(df_indices.keys())
df_t_mat = np.zeros([len(df_keys),len(df_keys)])
for i in range(len(df_keys)):
    for j in range(len(df_keys)):
        p_value = ss.ttest_ind(df['left'].iloc[df_indices[df_keys[i]]].values,df['left'].iloc[df_indices[df_keys[j]]].values)[1]
        if p_value<0.05:
            df_t_mat[i][j] = -1
        else:
            df_t_mat[i][j] = p_value
sns.heatmap(df_t_mat,xticklabels=df_keys,yticklabels=df_keys)
plt.show()


# 使用透视表来分析不同部门之间的离职率是否有关
# index表示行，columns表示列，aggfunc表示values的聚合方法。values越大表示离职率越高
piv_tb = pd.pivot_table(df,values='left',index=['promotion_last_5years','salary'],\
                        columns=['Work_accident'],aggfunc=np.mean)
print(piv_tb)
# 指定最大值为1，最小值为0
sns.heatmap(piv_tb,vmin=0,vmax=1,cmap=sns.color_palette('Reds',n_colors=256))
plt.show()
"""
"""
分组分析

# sns.barplot(x='salary',y='left',hue='department',data=df)
# plt.show()

sl_s = df['satisfaction_level']
sns.barplot(list(range(len(sl_s))),sl_s.sort_values())
# 从图中可见satisfaction_level属性的值具有两个拐点，因此可以在这两个拐点处进行切分分组
plt.show()
"""
"""
相关分析

# 计算相关系数时，会把离散值先去掉
sns.heatmap(df.corr(),vmin=-1,vmax=1,cmap=sns.color_palette('RdBu',n_colors=128))
plt.show()

s1 = pd.Series(['X1','X1','X2','X2','X2','X2'])
s2 = pd.Series(['Y1','Y1','Y1','Y2','Y2','Y2'])

# 熵
def getEntropy(s):
    if not isinstance(s,pd.core.series.Series):
        s = pd.Series(s)
    # 出现概率数组
    prt_ary = pd.groupby(s,by=s).count().values/float(len(s))
    return -(np.log2(prt_ary)*prt_ary).sum()
print('Entropy:',getEntropy(s2))

# 条件熵。已知s1的前提下，求s2的不确定性
def getCondEntropy(s1,s2):
    d = dict()
    for i in range(len(s1)):
        d[s1[i]] = d.get(s1[i],[])+[s2[i]]
    return sum([getEntropy(d[k])*len(d[k])/float(len(s1)) for k in d])
print('CondEntropy:',getCondEntropy(s2,s1))

# 互信息，信息增益
def getEntropyGain(s1,s2):
    return getEntropy(s2) - getCondEntropy(s1,s2)
print("EntropyGain:",getEntropyGain(s2,s1))

# 信息增益率
def getEntropyGainRatio(s1,s2):
    return getEntropyGain(s1,s2)/getEntropy(s2)
print("EntropyGainRatio:",getEntropyGainRatio(s2,s1))

# 衡量离散值的相关性
import math
def getDiscreteCorr(s1,s2):
    return getEntropyGain(s1,s2)/math.sqrt(getEntropy(s1)*getEntropy(s2))
print('DiscreteCorr:',getDiscreteCorr(s2,s1))

# 基尼系数
# 求概率平方和
def getProbSS(s):
    if not isinstance(s,pd.core.series.Series):
        s = pd.Series(s)
    # 出现概率数组
    prt_ary = pd.groupby(s,by=s).count().values/float(len(s))
    return sum(prt_ary**2)

def getGini(s1,s2):
    d = dict()
    for i in range(len(s1)):
        d[s1[i]] = d.get(s1[i], []) + [s2[i]]
    return 1 - sum([getProbSS(d[k])*len(d[k])/float(len(s1)) for k in d])
print('Gini:',getGini(s2,s1))
"""


"""
因子分析

from sklearn.decomposition import PCA
my_pca = PCA(n_components=7)
lower_mat = my_pca.fit_transform(df.drop(labels=['salary','department','left'],axis=1))
print('Ratio:',my_pca.explained_variance_ratio_)
sns.heatmap(pd.DataFrame(lower_mat).corr(),vmin=-1,vmax=1, cmap=sns.color_palette('ReBu',n_colors=128))
plt.show()
"""
