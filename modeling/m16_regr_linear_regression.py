#-*- coding: UTF-8 -*-
from modeling.m8_hr_preprocessing import hr_preprocessing
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import math

def regr_test(features,label):
    print('X',features)
    print('Y',label)
    # 线性回归
    # regr = LinearRegression()
    # 岭回归。这里使用岭回归可以控制参数的规模（参数间的差距），但影响不明显
    # regr = Ridge(alpha=1)
    # Lasso回归。对参数的影响很明显
    regr = Lasso(alpha=0.001)
    regr.fit(features.values,label.values)
    Y_pred = regr.predict(features.values)
    print('Coef:',regr.coef_)
    # MSE越小越好
    print('MSE:',mean_squared_error(label.values,Y_pred))
    print('RMSE:', math.sqrt(mean_squared_error(label.values, Y_pred)))
    print('MAE:', mean_absolute_error(label.values, Y_pred))
    print('R2:', r2_score(label.values, Y_pred))


if __name__ == '__main__':
    # 还可以对数据预处理中的各个参数进行调整
    # sl=False,le=False,npr=False,amh=False,tsc=False,wa=False,pl5=False,dp=False,slr=False,lower_d=False,ld_n=1
    features, label = hr_preprocessing()
    regr_test(features[['number_project','average_monthly_hours']],features['last_evaluation'])