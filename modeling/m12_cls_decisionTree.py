#-*- coding: UTF-8 -*-
from modeling.m8_hr_preprocessing import hr_preprocessing
from modeling.m9_dataset_split import dataset_split
from sklearn.metrics import accuracy_score,recall_score,f1_score
# 保存模型
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier,export_graphviz
# 画出决策树
import pydotplus
from sklearn.externals.six import StringIO

# 此案例中，决策树的效果好过KNN

def cls_decisionTree(X_train,X_validation,X_test,Y_train,Y_validation,Y_test,features_name):
    models = []
    models.append(('DecisionTreeGini',DecisionTreeClassifier()))
    # 信息增益
    models.append(('DecisionTreeEntropy',DecisionTreeClassifier(criterion='entropy')))
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

            """
            # 输出决策树图片到PDF文件
            dot_data = StringIO()
            export_graphviz(clf,out_file=dot_data,feature_names=features_name,
                                       class_names=['NOLeft','Left'],filled=True,rounded=True,special_characters=True)
            graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
            graph.write_pdf('graph_cls_decisionTree2.pdf')
            """

if __name__ == '__main__':
    # 还可以对数据预处理中的各个参数进行调整
    # sl=False,le=False,npr=False,amh=False,tsc=False,wa=False,pl5=False,dp=False,slr=False,lower_d=False,ld_n=1
    features, label = hr_preprocessing()
    # 获取特征名称
    features_name = features.columns.values
    X_train,X_validation,X_test,Y_train,Y_validation,Y_test = dataset_split(features, label)
    cls_decisionTree(X_train,X_validation,X_test,Y_train,Y_validation,Y_test,features_name)