#-*- coding: UTF-8 -*-
from modeling.m8_hr_preprocessing import hr_preprocessing
from modeling.m9_dataset_split import dataset_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,recall_score,f1_score
from sklearn.externals import joblib

def cls_KNN(X_train,X_validation,X_test,Y_train,Y_validation,Y_test):
    # 可以调整参数n_neighbors，然后观察效果。这里发现等于3的效果比等于5要好
    knn_clf = KNeighborsClassifier(n_neighbors=3)
    # knn_clf_n5 = KNeighborsClassifier(n_neighbors=5)
    knn_clf.fit(X_train,Y_train)
    # knn_clf_n5.fit(X_train,Y_train)
    Y_predict = knn_clf.predict(X_validation)
    # Y_predict_n5 = knn_clf_n5.predict(X_validation)
    # print('n_neighbors=3:')
    print('ACC:',accuracy_score(Y_validation,Y_predict))
    print('REC:', recall_score(Y_validation, Y_predict))
    print('F-Score:', f1_score(Y_validation, Y_predict))
    # print('n_neighbors=5:')
    # print('ACC:', accuracy_score(Y_validation, Y_predict_n5))
    # print('REC:', recall_score(Y_validation, Y_predict_n5))
    # print('F-Score:', f1_score(Y_validation, Y_predict_n5))

    print('Test:')
    Y_predict = knn_clf.predict(X_test)
    print('ACC:', accuracy_score(Y_test, Y_predict))
    print('REC:', recall_score(Y_test, Y_predict))
    print('F-Score:', f1_score(Y_test, Y_predict))

    # 保存模型
    joblib.dump(knn_clf,'m10_cls_knn')

    # 加载模型，然后进行使用
    knn_clf2 = joblib.load('m10_cls_knn')
    print('Test2:')
    Y_pred = knn_clf2.predict(X_test)
    print('ACC:', accuracy_score(Y_test, Y_pred))
    print('REC:', recall_score(Y_test, Y_pred))
    print('F-Score:', f1_score(Y_test, Y_pred))


if __name__ == '__main__':
    # 还可以对数据预处理中的各个参数进行调整
    # sl=False,le=False,npr=False,amh=False,tsc=False,wa=False,pl5=False,dp=False,slr=False,lower_d=False,ld_n=1
    features, label = hr_preprocessing()
    X_train,X_validation,X_test,Y_train,Y_validation,Y_test = dataset_split(features, label)
    cls_KNN(X_train,X_validation,X_test,Y_train,Y_validation,Y_test)