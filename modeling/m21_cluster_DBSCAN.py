#-*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles,make_blobs,make_moons
from sklearn.cluster import DBSCAN

# 定义1000个样本点
n_samples = 1000
# 噪声比例为0.05,factor表示小圆与大圆之间的间距
circles = make_circles(n_samples=n_samples,factor=0.5,noise=0.05)
# random_state使得8次产生的随机值都是一样的，以免产生位置上的变化
blobs = make_blobs(n_samples=n_samples,random_state=8,center_box=(-1,1),cluster_std=0.1)
moons = make_moons(n_samples=n_samples,noise=0.05)
# random_data 为2维数据。None表示不使用标注
random_data = np.random.rand(n_samples,2),None
colors = 'bgrcmyk'
# 总共有四类数据。前三类数据都带有标注，这里并不需要使用标注。random_data的标注为None
data = [circles,moons,blobs,random_data]
# (模型名，模型实体) 即下面的 clt。min_samples即最小点数MinPts、eps即半径E
models = [('None',None),('DBSCAN',DBSCAN(min_samples=3,eps=0.2))]
f = plt.figure()
for inx,clt in enumerate(models):
    clt_name,clt_entity = clt
    for i,dataset in enumerate(data):
        # 特征X，标注Y
        X,Y = dataset
        # 若模型实体为空，则模型结果为0
        if not clt_entity:
            clt_result = [0 for item in range(len(X))]
        else:
            clt_entity.fit(X)
            clt_result = clt_entity.labels_.astype(np.int)
        # inx*len(data)代表第几行，i+1代表第几个位置，该参数默认从1开始
        f.add_subplot(len(models),len(data),inx*len(data)+i+1)
        # 散点图。特征X是二维的
        plt.title(clt_name)
        [plt.scatter(X[p,0],X[p,1],color=colors[clt_result[p]]) for p in range(len(X))]
plt.show()