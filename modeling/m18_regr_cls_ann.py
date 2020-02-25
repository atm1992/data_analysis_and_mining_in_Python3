#-*- coding: UTF-8 -*-
from modeling.m8_hr_preprocessing import hr_preprocessing
from modeling.m9_dataset_split import dataset_split
from sklearn.metrics import accuracy_score,recall_score,f1_score
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc,roc_auc_score

def regr_cls_ann(X_train,X_validation,X_test,Y_train,Y_validation,Y_test):
    mdl = Sequential()
    mdl.add(Dense(50, input_dim=len(X_train[0])))
    mdl.add(Activation('sigmoid'))
    mdl.add(Dense(2))
    mdl.add(Activation('softmax'))
    sgd = SGD(lr=0.1)
    # SGD收敛速度太慢，因此这里使用的是其变种adam
    mdl.compile(loss='mse', optimizer='adam')
    mdl.fit(X_train,np.array([[0,1] if i==1 else [1,0] for i in Y_train]),nb_epoch=1000,batch_size=8999)
    xy_lst = [(X_train, Y_train), (X_validation, Y_validation), (X_test, Y_test)]
    f = plt.figure()
    for i in range(len(xy_lst)):
        X_part = xy_lst[i][0]
        Y_part = xy_lst[i][1]
        Y_pred = mdl.predict(X_part)
        # Y_pred为二维数组，子数组中有两个元素，第一个为预测为0的概率，第二个为预测为1的概率
        print(Y_pred)
        Y_pred = np.array(Y_pred[:,1]).reshape((1,-1))[0]
        # print(i)
        # print('ANN', '-ACC:', accuracy_score(Y_part, Y_pred))
        # print('ANN', '-REC:', recall_score(Y_part, Y_pred))
        # print('ANN', '-F-Score:', f1_score(Y_part, Y_pred))
        f.add_subplot(1,3,i+1)
        fpr,tpr,threshold = roc_curve(Y_part,Y_pred)
        plt.plot(fpr,tpr)
        # auc 与 roc_auc_score是等效的，结果相同
        print('NN','AUC',auc(fpr,tpr))
        print('NN','auc_score',roc_auc_score(Y_part,Y_pred))
    plt.show()

if __name__ == '__main__':
    # 还可以对数据预处理中的各个参数进行调整
    # sl=False,le=False,npr=False,amh=False,tsc=False,wa=False,pl5=False,dp=False,slr=False,lower_d=False,ld_n=1
    features, label = hr_preprocessing()
    X_train,X_validation,X_test,Y_train,Y_validation,Y_test = dataset_split(features, label)
    regr_cls_ann(X_train,X_validation,X_test,Y_train,Y_validation,Y_test)