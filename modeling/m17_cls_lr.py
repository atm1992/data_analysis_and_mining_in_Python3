#-*- coding: UTF-8 -*-
from modeling.m8_hr_preprocessing import hr_preprocessing
from modeling.m9_dataset_split import dataset_split
from sklearn.metrics import accuracy_score,recall_score,f1_score
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression

# 逻辑回归、线性回归不适用于本案例，因为本案例的数据集并不是线性可分的
# 若要处理非线性问题，则需将数据先映射到高维空间，使其线性可分，类似于SVM

def cls_lr(X_train,X_validation,X_test,Y_train,Y_validation,Y_test):
    models = []
    models.append(('LR',LogisticRegression(C=1000,tol=1e-10,solver='sag',max_iter=10000)))
    for clf_name,clf in models:
        clf.fit(X_train,Y_train)
        xy_lst = [(X_train,Y_train),(X_validation,Y_validation),(X_test,Y_test)]
        for i in range(len(xy_lst)):
            X_part = xy_lst[i][0]
            Y_part = xy_lst[i][1]
            Y_pred = clf.predict(X_part)
            print(i)
            print(clf_name,'-ACC:', accuracy_score(Y_part, Y_pred))
            print(clf_name,'-REC:', recall_score(Y_part, Y_pred))
            print(clf_name,'-F-Score:', f1_score(Y_part, Y_pred))

if __name__ == '__main__':
    # 还可以对数据预处理中的各个参数进行调整
    # sl=False,le=False,npr=False,amh=False,tsc=False,wa=False,pl5=False,dp=False,slr=False,lower_d=False,ld_n=1
    features, label = hr_preprocessing()
    X_train,X_validation,X_test,Y_train,Y_validation,Y_test = dataset_split(features, label)
    cls_lr(X_train,X_validation,X_test,Y_train,Y_validation,Y_test)