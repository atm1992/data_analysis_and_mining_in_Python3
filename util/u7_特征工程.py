#-*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
import scipy.stats as ss

"""
数据清洗

df = pd.DataFrame({
    'A': ['a0', 'a1', 'a1', 'a2', 'a3', 'a4'],
    'B': ['b0', 'b1', 'b2', 'b2', 'b3', None],
    'C': [1, 2, None, 3, 4, 5],
    'D': [0.1, 10.2, 11.4, 8.9, 9.1, 12],
    'E': [10, 19, 32, 25, 8, None],
    'F': ['f0', 'f1', 'g2', 'f3', 'f4', 'f5']
})
print(df)
print(df.isnull())
print(df.dropna())    # 删除所有存在空值的行
print(df.dropna(subset=['B']))    # 删除'B'列中值为空所在的行
print(df.duplicated(['A']))
print(df.duplicated(['A', 'B']))    # 显示'A'、'B'两列同时重复的行
print(df.drop_duplicates(['A'], keep='first'))
print(df.fillna('b*'))
print(df.fillna(df['E'].mean()))
print(df['E'].interpolate())    # 插值
upper_q = df['D'].quantile(0.75)
lower_q = df['D'].quantile(0.25)
q_int = upper_q - lower_q
k = 1.5
print('过滤掉边界值以外的异常值:\n', df[df['D'] > lower_q - k*q_int][df['D'] < upper_q + k*q_int])
print(df[[True if item.startswith('f') else False for item in list(df['F'].values)]])
"""


"""
特征预处理 —— 特征选择

# D属性为标注，取值为0或1
df = pd.DataFrame({'A':ss.norm.rvs(size=10),'B':ss.norm.rvs(size=10)
                   ,'C':ss.norm.rvs(size=10),'D':np.random.randint(low=0,high=2,size=10)})
print(df)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
X = df.loc[:,['A','B','C']]
Y = df.loc[:,'D']

from sklearn.feature_selection import SelectKBest,RFE,SelectFromModel

skb = SelectKBest(k=2)      # 保留在指定score_func下的分值最高的两个特征(属性)
skb.fit(X,Y)
print(skb.transform(X))

# n_features_to_select为保留的特征个数，step表示每次去掉的特征个数
rfe = RFE(estimator=SVR(kernel='linear'),n_features_to_select=2,step=1)
# 保留结果未必与上面的skb一致
print(rfe.fit_transform(X,Y))

# threshold表示特征重要性因子，低于这个值就会被去掉。最终保留的特征个数不确定，阈值越低，所保留的特征个数就越多
sfm = SelectFromModel(estimator=DecisionTreeRegressor(),threshold=0.1)
print(sfm.fit_transform(X,Y))
"""



"""
分箱

lst = [6,8,10,15,16,24,25,40,67]
# 等频（等身）分箱。用区间值来代替实际数据
print(pd.qcut(lst,q=3))
print(pd.qcut(lst,q=3,labels=['low','medium','high']))

# 等距（等宽）分箱。用区间值来代替实际数据
print(pd.cut(lst,bins=3))
print(pd.cut(lst,bins=3,labels=['low','medium','high']))
"""


"""
# 归一化
from sklearn.preprocessing import MinMaxScaler
print(MinMaxScaler().fit_transform(np.array([1,4,10,15,21]).reshape(-1,1)))

# 标准化
from sklearn.preprocessing import StandardScaler
print(StandardScaler().fit_transform(np.array([1,1,1,1,0,0,0,0]).reshape(-1,1)))
print(StandardScaler().fit_transform(np.array([1,0,0,0,0,0,0,0]).reshape(-1,1)))
"""


"""
数值化

from sklearn.preprocessing import LabelEncoder,OneHotEncoder

print(LabelEncoder().fit_transform(np.array(['Down','Up','Up','Down']).reshape(-1,1)))
print(LabelEncoder().fit_transform(np.array(['Low','Medium','High','Medium','Low']).reshape(-1,1)))

lb_encoder = LabelEncoder()
lb_tran_f = lb_encoder.fit_transform(np.array(['Red','Yellow','Blue','Green']))
oht_encoder = OneHotEncoder().fit(lb_tran_f.reshape(-1,1))
print(oht_encoder.transform(lb_encoder.transform(np.array(['Yellow','Blue','Green','Green','Red'])).reshape(-1,1)).toarray())
"""


"""
正则化

from sklearn.preprocessing import Normalizer
# 对行进行正则化
print(Normalizer(norm='l1').fit_transform(np.array([[1,1,3,-1,2]])))
print(Normalizer(norm='l2').fit_transform(np.array([[1,1,3,-1,2]])))
"""

"""
特征降维 —— LDA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

X = np.array([[-1,-1],[-2,-1],[-3,-2],[1,1],[2,1],[3,2]])
Y = np.array([1,1,1,2,2,2])
# n_components=1 表示降到一维
print(LinearDiscriminantAnalysis(n_components=1).fit_transform(X,Y))

# LDA降维后，可以当作一个分类器来使用，即 Fisher分类器
clf = LinearDiscriminantAnalysis(n_components=1).fit(X,Y)
print(clf.predict([[0.8,1]]))
 """