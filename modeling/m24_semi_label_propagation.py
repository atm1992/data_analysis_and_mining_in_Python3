#-*- coding: UTF-8 -*-
import numpy as np
from sklearn import datasets
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import accuracy_score,recall_score,f1_score

iris = datasets.load_iris()
labels = np.copy(iris.target)
print(len(labels))
# 生成len(iris.target)个0~1之间的随机数
random_unlabeled_points = np.random.rand(len(iris.target))
# 若随机数小于0.3，则赋值为1；若随机数大于等于0.3，则赋值为0
random_unlabeled_points = random_unlabeled_points<0.5
# 使用Y来保存random_unlabeled_points为1的样本的真实标注
Y = labels[random_unlabeled_points]
# 将random_unlabeled_points为1的样本的标注替换为-1，表示没有打标注
labels[random_unlabeled_points] = -1
print('unlabels',list(labels).count(-1))
label_prop_model = LabelPropagation().fit(iris.data,labels)
Y_pred = label_prop_model.predict(iris.data)
# 获取random_unlabeled_points为1的样本的预测标注
Y_pred = Y_pred[random_unlabeled_points]
# 计算指标
print('ACC',accuracy_score(Y,Y_pred))
print('REC',recall_score(Y,Y_pred,average='micro'))
print('F1',f1_score(Y,Y_pred,average='micro'))