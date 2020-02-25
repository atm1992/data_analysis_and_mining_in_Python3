#-*- coding: UTF-8 -*-
from modeling.m8_hr_preprocessing import hr_preprocessing
from modeling.m9_dataset_split import dataset_split
from sklearn.metrics import accuracy_score,recall_score,f1_score
from sklearn.externals import joblib
from sklearn.svm import SVC

def cls_svm(X_train,X_validation,X_test,Y_train,Y_validation,Y_test):
    models = []
    # 参数C表示错分点的惩罚度，不过计算时间会变长，但是分类效果比默认的惩罚度要好
    models.append(('SVM',SVC(C=10000)))
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
    cls_svm(X_train,X_validation,X_test,Y_train,Y_validation,Y_test)