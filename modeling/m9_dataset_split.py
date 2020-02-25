#-*- coding: UTF-8 -*-
# sklearn中没有哪个方法可以一步直接切分出训练集、验证集、测试集，因此这里通过使用两次train_test_split来实现
from sklearn.model_selection import train_test_split
from modeling.m8_hr_preprocessing import hr_preprocessing

def dataset_split(features,label):
    f_v = features.values
    l_v = label.values
    # 返回训练集部分(80%) 与 验证集部分(20%)
    X_tt,X_validation,Y_tt,Y_validation= train_test_split(f_v,l_v,test_size=.2)
    # 返回最终的训练集和测试集。训练集占原有数据的60%，测试集占原有数据的20%，验证集占原有数据的20%
    X_train,X_test,Y_train,Y_test = train_test_split(X_tt,Y_tt,test_size=0.25)
    # print(len(X_train),len(X_validation),len(X_test))
    return X_train,X_validation,X_test,Y_train,Y_validation,Y_test

if __name__ == '__main__':
    features, label = hr_preprocessing()
    dataset_split(features, label)