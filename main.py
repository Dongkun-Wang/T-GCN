# -*- coding: utf-8 -*-
import pickle as pkl
import tensorflow.compat.v1 as tf
import pandas as pd
import numpy as np
import math
import os
from utils import evaluation
from datareader import preprocess_data, load_data
from tgcn import TGCN_Cell
from configs import args
from keras import layers

# from gru import GRUCell
# tf.disable_v2_behavior()
from visualization import plot_result, plot_error
# import matplotlib.pyplot as plt
import time

time_start = time.time()

model_name = args.model_name
city = args.city
train_rate = args.train_rate
seq_len = args.seq_len
output_dim = pre_len = args.pre_len
batch_size = args.batch_size
lr = args.learning_rate
training_epoch = args.training_epoch
gru_units = args.gru_units

###### load data
data, adj = load_data(city)

time_len = data.shape[0]
num_nodes = data.shape[1]
data = np.mat(data, dtype=np.float32)

#### normalization
data = data / np.max(data)
trainX, trainY, testX, testY = preprocess_data(data, time_len, train_rate, seq_len, pre_len)

batch_number = int(trainX.shape[0] / batch_size)
training_data_count = len(trainX)


def TGCN(_X, _weights, _biases):
    ###
    cell_1 = TGCN_Cell(gru_units, adj, num_nodes=num_nodes)
    cell = tf.keras.layers.StackedRNNCells([cell_1])
    _X = tf.unstack(_X, axis=1)
    outputs, states = tf.nn.static_rnn(cell, _X, dtype=tf.float32)
    # outputs, states = layers.RNN(cell, _X, unroll=True)
    m = []
    for i in outputs:
        o = tf.reshape(i, shape=[-1, num_nodes, gru_units])
        o = tf.reshape(o, shape=[-1, gru_units])
        m.append(o)
    last_output = m[-1]
    output = tf.matmul(last_output, _weights['out']) + _biases['out']
    output = tf.reshape(output, shape=[-1, num_nodes, pre_len])
    output = tf.transpose(output, perm=[0, 2, 1])
    output = tf.reshape(output, shape=[-1, num_nodes])
    return output, m, states


###### placeholders ######
inputs = tf.placeholder(tf.float32, shape=[None, seq_len, num_nodes])
labels = tf.placeholder(tf.float32, shape=[None, pre_len, num_nodes])

# Graph weights output FC layer
weights = {
    'out': tf.Variable(tf.random_normal([gru_units, pre_len], mean=1.0), name='weight_o')}
biases = {
    'out': tf.Variable(tf.random_normal([pre_len]), name='bias_o')}

if model_name == 'T-GCN':
    pred, _, _ = TGCN(inputs, weights, biases)

y_pred = pred

###### optimizer ######
lambda_loss = 0.0015
Lreg = lambda_loss * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
label = tf.reshape(labels, [-1, num_nodes])
##loss
loss = tf.reduce_mean(tf.nn.l2_loss(y_pred - label) + Lreg)
##rmse
error = tf.sqrt(tf.reduce_mean(tf.square(y_pred - label)))
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

###### Initialize session ######
variables = tf.global_variables()
saver = tf.train.Saver(tf.global_variables())
# sess = tf.Session()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())

out = 'result/%s' % (model_name)
# out = 'out/%s_%s'%(model_name,'perturbation')
path = 'result/%s_%s_lr%r_batch%r_unit%r_seq%r_pre%r_epoch%r' % (
    model_name, city, lr, batch_size, gru_units, seq_len, pre_len, training_epoch)
# path = os.path.join(out, path1)
if not os.path.exists(path):
    os.makedirs(path)


###### evaluation ######



x_axe, batch_loss, batch_rmse, batch_pred = [], [], [], []
test_loss, test_rmse, test_mae, test_acc, test_r2, test_var, test_pred = [], [], [], [], [], [], []

for epoch in range(training_epoch):
    for m in range(batch_number):
        mini_batch = trainX[m * batch_size: (m + 1) * batch_size]
        mini_label = trainY[m * batch_size: (m + 1) * batch_size]
        _, loss1, rmse1, train_output = sess.run([optimizer, loss, error, y_pred],
                                                 feed_dict={inputs: mini_batch, labels: mini_label})
        batch_loss.append(loss1)
        batch_rmse.append(rmse1 * max_value)

    # Test completely at every epoch
    loss2, rmse2, test_output = sess.run([loss, error, y_pred],
                                         feed_dict={inputs: testX, labels: testY})
    test_label = np.reshape(testY, [-1, num_nodes])
    rmse, mae, acc, r2_score, var_score = evaluation(test_label, test_output)
    test_label1 = test_label * max_value
    test_output1 = test_output * max_value
    test_loss.append(loss2)
    test_rmse.append(rmse * max_value)
    test_mae.append(mae * max_value)
    test_acc.append(acc)
    test_r2.append(r2_score)
    test_var.append(var_score)
    test_pred.append(test_output1)

    print('Epoch{}:'.format(epoch + 1),
          'train_rmse:{:.4}\t'.format(batch_rmse[-1]),
          'test_loss:{:.4}\n'.format(loss2),
          '\t \t \t test_rmse:{:.4}\t'.format(rmse),
          'test_acc:{:.4}'.format(acc))

    if epoch % 100 == 0:
        saver.save(sess, path + '/model_100/TGCN_pre_%r' % epoch, global_step=epoch)

time_end = time.time()
print(time_end - time_start, 's')

# visualization
b = int(len(batch_rmse) / batch_number)
batch_rmse1 = [i for i in batch_rmse]
train_rmse = [(sum(batch_rmse1[i * batch_number:(i + 1) * batch_number]) / batch_number) for i in range(b)]
batch_loss1 = [i for i in batch_loss]
train_loss = [(sum(batch_loss1[i * batch_number:(i + 1) * batch_number]) / batch_number) for i in range(b)]

index = test_rmse.index(np.min(test_rmse))
test_result = test_pred[index]
var = pd.DataFrame(test_result)
var.to_csv(path + '/test_result.csv', index=False, header=False)
# plot_result(test_result,test_label1,path)
# plot_error(train_rmse,train_loss,test_rmse,test_acc,test_mae,path)

print('min_rmse:%r' % (np.min(test_rmse)),
      'min_mae:%r' % (test_mae[index]),
      'max_acc:%r' % (test_acc[index]),
      'r2:%r' % (test_r2[index]),
      'var:%r' % test_var[index])
