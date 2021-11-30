# -*- coding: utf-8 -*-
# @：baselines included:
# (1) History Average model (HA)
# (2) Autoregressive Integrated Moving Average model (ARIMA)
# (3) Support Vector Regression model (SVR)
# (4) Graph Convolutional Network model (GCN)
# (5) Gated Recurrent Unit model (GRU)

import pandas as pd
import numpy as np
from utils import evaluation
from datareader import *
from sklearn.svm import SVR
from statsmodels.tsa.arima_model import ARIMA


path = r'data/LA_speed.csv'
data = pd.read_csv(path)

time_len = data.shape[0]
num_nodes = data.shape[1]
train_rate = 0.8
seq_len = 12
pre_len = 3
trainX, trainY, testX, testY, max_value = preprocess_data(data, time_len, train_rate, seq_len, pre_len)
method = 'HA'  ####HA or SVR or ARIMA

########### HA #############
if method == 'HA':
    result = []
    for i in range(len(testX)):
        a = np.array(testX[i])
        tempResult = []

        a1 = np.mean(a, axis=0)
        tempResult.append(a1)
        a = a[1:]
        a = np.append(a, [a1], axis=0)
        a1 = np.mean(a, axis=0)
        tempResult.append(a1)
        a = a[1:]
        a = np.append(a, [a1], axis=0)
        a1 = np.mean(a, axis=0)
        tempResult.append(a1)

        result.append(tempResult)
    result1 = np.array(result)
    result1 = np.reshape(result1, [-1, num_nodes])
    testY1 = np.array(testY)
    testY1 = np.reshape(testY1, [-1, num_nodes])
    rmse, mae, accuracy, r2, var = evaluation(testY1, result1)
    print('HA_rmse:%r' % rmse,
          'HA_mae:%r' % mae,
          'HA_acc:%r' % accuracy,
          'HA_r2:%r' % r2,
          'HA_var:%r' % var)

############ SVR #############
if method == 'SVR':
    total_rmse, total_mae, total_acc, result = [], [], [], []
    for i in range(num_nodes):
        data1 = np.mat(data)
        a = data1[:, i]
        a_X, a_Y, t_X, t_Y = preprocess_data(a, time_len, train_rate, seq_len, pre_len)
        a_X = np.array(a_X)
        a_X = np.reshape(a_X, [-1, seq_len])
        a_Y = np.array(a_Y)
        a_Y = np.reshape(a_Y, [-1, pre_len])
        a_Y = np.mean(a_Y, axis=1)
        t_X = np.array(t_X)
        t_X = np.reshape(t_X, [-1, seq_len])
        t_Y = np.array(t_Y)
        t_Y = np.reshape(t_Y, [-1, pre_len])

        svr_model = SVR(kernel='linear')
        svr_model.fit(a_X, a_Y)
        pre = svr_model.predict(t_X)
        pre = np.array(np.transpose(np.mat(pre)))
        pre = pre.repeat(pre_len, axis=1)
        result.append(pre)
    result1 = np.array(result)
    result1 = np.reshape(result1, [num_nodes, -1])
    result1 = np.transpose(result1)
    testY1 = np.array(testY)

    testY1 = np.reshape(testY1, [-1, num_nodes])
    total = np.mat(total_acc)
    total[total < 0] = 0
    rmse1, mae1, acc1, r2, var = evaluation(testY1, result1)
    print('SVR_rmse:%r' % rmse1,
          'SVR_mae:%r' % mae1,
          'SVR_acc:%r' % acc1,
          'SVR_r2:%r' % r2,
          'SVR_var:%r' % var)

######## ARIMA #########
if method == 'ARIMA':
    rng = pd.date_range('1/3/2012', periods=2016, freq='15min')
    a1 = pd.DatetimeIndex(rng)
    data.index = a1
    num = data.shape[1]
    rmse, mae, acc, r2, var, pred, ori = [], [], [], [], [], [], []
    for i in range(156):
        ts = data.iloc[:, i]
        ts_log = np.log(ts)
        ts_log = np.array(ts_log, dtype=np.float)
        where_are_inf = np.isinf(ts_log)
        ts_log[where_are_inf] = 0
        ts_log = pd.Series(ts_log)
        ts_log.index = a1
        model = ARIMA(ts_log, order=[1, 0, 0])
        properModel = model.fit()
        predict_ts = properModel.predict(4, dynamic=True)
        log_recover = np.exp(predict_ts)
        ts = ts[log_recover.index]
        er_rmse, er_mae, er_acc, r2_score, var_score = evaluation(ts, log_recover)
        rmse.append(er_rmse)
        mae.append(er_mae)
        acc.append(er_acc)
        r2.append(r2_score)
        var.append(var_score)
    #    for i in range(109,num):
    #        ts = data.iloc[:,i]
    #        ts_log=np.log(ts)
    #        ts_log=np.array(ts_log,dtype=np.float)
    #        where_are_inf = np.isinf(ts_log)
    #        ts_log[where_are_inf] = 0
    #        ts_log = pd.Series(ts_log)
    #        ts_log.index = a1
    #        model = ARIMA(ts_log,order=[1,1,1])
    #        properModel = model.fit(disp=-1, method='css')
    #        predict_ts = properModel.predict(2, dynamic=True)
    #        log_recover = np.exp(predict_ts)
    #        ts = ts[log_recover.index]
    #        er_rmse,er_mae,er_acc,r2_score,var_score = evaluation(ts,log_recover)
    #        rmse.append(er_rmse)
    #        mae.append(er_mae)
    #        acc.append(er_acc)
    #        r2.append(r2_score)
    #        var.append(var_score)
    acc1 = np.mat(acc)
    acc1[acc1 < 0] = 0
    print('arima_rmse:%r' % (np.mean(rmse)),
          'arima_mae:%r' % (np.mean(mae)),
          'arima_acc:%r' % (np.mean(acc1)),
          'arima_r2:%r' % (np.mean(r2)),
          'arima_var:%r' % (np.mean(var)))
