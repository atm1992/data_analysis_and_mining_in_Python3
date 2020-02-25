# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import Normalizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA


# 8-HR表的特征预处理

# sl:satisfaction_level —— False:MinMaxScaler；True:StandardScaler
# le:last_evaluation —— False:MinMaxScaler；True:StandardScaler
# npr:number_project —— False:MinMaxScaler；True:StandardScaler
# amh:average_monthly_hours —— False:MinMaxScaler；True:StandardScaler
# tsc:time_spend_company —— False:MinMaxScaler；True:StandardScaler
# wa:Work_accident —— False:MinMaxScaler；True:StandardScaler
# pl5:promotion_last_5years —— False:MinMaxScaler；True:StandardScaler
# dp:department —— False:LabelEncoder；True:OneHotEncoder
# slr:salary —— False:LabelEncoder；True:OneHotEncoder
# lower_d —— 是否需要降维
# ld_n —— 如果需要降维，则最终需要保留多少维
def hr_preprocessing(sl=False, le=False, npr=False, amh=False, tsc=False, wa=False, pl5=False, dp=False, slr=False,lower_d=False, ld_n=1):
    df = pd.read_csv("../data/HR.csv")

    # 1、清洗数据
    df = df.dropna(subset=['satisfaction_level', 'last_evaluation'])
    df = df[df['satisfaction_level'] <= 1][df['salary'] != 'nme']
    # 2、得到标注
    label = df['left']
    # axis=1 表示按列删除
    df = df.drop('left', axis=1)
    # 3、特征选择
    # 可以剔除与标注left相关性不大的属性，如last_evaluation
    # 4、特征处理
    scaler_lst = [sl, le, npr, amh, tsc, wa, pl5]
    column_lst = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_monthly_hours',
                  'time_spend_company', 'Work_accident', 'promotion_last_5years']
    for i in range(len(scaler_lst)):
        if not scaler_lst[i]:
            df[column_lst[i]] = MinMaxScaler().fit_transform(df[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[0]
        else:
            df[column_lst[i]] = StandardScaler().fit_transform(df[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[0]
    scaler_lst = [slr, dp]
    column_lst = ['salary', 'department']
    for i in range(len(scaler_lst)):
        if not scaler_lst[i]:
            if column_lst[i] == 'salary':
                df[column_lst[i]] = [map_salary(s) for s in df['salary'].values]
            else:
                df[column_lst[i]] = LabelEncoder().fit_transform(df[column_lst[i]])
            # LabelEncoder之后，再进行归一化
            df[column_lst[i]] = MinMaxScaler().fit_transform(df[column_lst[i]].values.reshape(-1, 1)).reshape(1, -1)[0]
        else:
            # pandas中的OneHotEncoder
            df = pd.get_dummies(df, columns=[column_lst[i]])
    if lower_d:
        return PCA(n_components=ld_n).fit_transform(df.values)
    return df, label


d = {'low': 0, 'medium': 1, 'high': 2}


def map_salary(s):
    return d.get(s, 0)


if __name__ == '__main__':
    features, label = hr_preprocessing()
