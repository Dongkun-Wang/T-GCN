# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def load_data(city):
    adj_path = 'data/{}_adj.csv'.format(city)
    speed_path = 'data/{}_speed.csv'.format(city)
    adj = np.mat(pd.read_csv(adj_path, header=None))
    speed = pd.read_csv(speed_path)
    return speed, adj


def preprocess_data(data, time_len, rate, seq_len, pre_len):
    max_value = np.max(data)
    data = data / np.max(data)
    train_size = int(time_len * rate)
    train_data = data[0:train_size]
    test_data = data[train_size:time_len]

    trainX, trainY, testX, testY = [], [], [], []
    for i in range(len(train_data) - seq_len - pre_len):
        a = train_data[i: i + seq_len + pre_len]
        trainX.append(a[0: seq_len])
        trainY.append(a[seq_len: seq_len + pre_len])
    for i in range(len(test_data) - seq_len - pre_len):
        b = test_data[i: i + seq_len + pre_len]
        testX.append(b[0: seq_len])
        testY.append(b[seq_len: seq_len + pre_len])

    trainX = np.array(trainX)
    trainY = np.array(trainY)
    testX = np.array(testX)
    testY = np.array(testY)
    return trainX, trainY, testX, testY, max_value
