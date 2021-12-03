# -*- coding: utf-8 -*-
import pickle as pkl
import tensorflow.compat.v1 as tf
import os
from utils import evaluation
from datareader import *
from tgcn import Model
from configs import args
from keras import layers
# from gru import GRUCell
from visualization import plot_result, plot_error
# import matplotlib.plot as plt
import time
tf.disable_v2_behavior()

# if __name__ == "__main__":
time_start = time.time()
model_name = args.model_name
city = args.city
train_rate = args.train_rate
seq_len = args.seq_len
pre_len = args.pre_len
batch_size = args.batch_size
learning_rate = args.learning_rate
training_epoch = args.training_epoch
gru_units = args.gru_units

# load data
data, adjency = load_data(city)
time_len = data.shape[0]
num_nodes = data.shape[1]
data = np.mat(data, dtype=np.float32)
# normalization
trainX, trainY, testX, testY, max_value = \
    preprocess_data(data, time_len, train_rate, seq_len, pre_len)
batch_number = len(trainX) // batch_size

# placeholders
inputs = tf.placeholder(tf.float32, shape=[None, seq_len, num_nodes])
labels = tf.placeholder(tf.float32, shape=[None, pre_len, num_nodes])

y_pred = Model(inputs, gru_units, adjency, num_nodes, pre_len)
label = tf.reshape(labels, [-1, num_nodes])

# loss function
# lambda_loss = 0.0015
# L_reg = lambda_loss * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
# loss = tf.reduce_mean(tf.nn.l2_loss(y_pred - label) + L_reg)
loss = tf.reduce_mean(tf.nn.l2_loss(y_pred - label))
error = tf.sqrt(tf.reduce_mean(tf.square(y_pred - label)))
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Initialize session
variables = tf.global_variables()
saver = tf.train.Saver(tf.global_variables())
gpu_options = tf.GPUOptions()
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(tf.global_variables_initializer())

path = 'result/%s_%s_lr%r_batch%r_unit%r_seq%r_pre%r_epoch%r' % (
    model_name, city, learning_rate, batch_size, gru_units, seq_len, pre_len, training_epoch)
if not os.path.exists(path):
    os.makedirs(path)

x_axe, batch_loss, batch_rmse, batch_pred = [], [], [], []
test_loss, test_rmse, test_mae, test_acc, test_r2, test_var, test_pred = [], [], [], [], [], [], []

for epoch in range(training_epoch):
    for m in range(batch_number):
        batch_inputs = trainX[m * batch_size: (m + 1) * batch_size]
        batch_labels = trainY[m * batch_size: (m + 1) * batch_size]
        _, train_loss, train_rmse, train_output = sess.run([optimizer, loss, error, y_pred],
                                                           feed_dict={inputs: batch_inputs, labels: batch_labels})
        batch_loss.append(train_loss)
        batch_rmse.append(train_rmse * max_value)

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

    print('Epoch{}:\t'.format(epoch + 1), 'train_rmse:{:.4}\t'.format(batch_rmse[-1]),
          'test_loss:{:.4}\t'.format(loss2), 'test_rmse:{:.4}\t'.format(rmse), 'test_acc:{:.4}\n'.format(acc))

    if epoch % 100 == 0:
        saver.save(sess, path + '/model_100/T-GCN', global_step=epoch)

time_end = time.time()
print(time_end - time_start, 's')

# visualization
b = len(batch_rmse) // batch_number
batch_rmse1 = [i for i in batch_rmse]
final_train_rmse = [(sum(batch_rmse1[i * batch_number:(i + 1) * batch_number])
                     / batch_number) for i in range(b)]
batch_loss1 = [i for i in batch_loss]
final_train_loss = [(sum(batch_loss1[i * batch_number:(i + 1) * batch_number])
                     / batch_number) for i in range(b)]

index = test_rmse.index(np.min(test_rmse))
test_result = test_pred[index]
var = pd.DataFrame(test_result)
var.to_csv(path + '/test_result.csv', index=False, header=False)
# plot_result(test_result,test_label1,path)
# plot_error(final_train_rmse,final_train_loss,test_rmse,test_acc,test_mae,path)

print('min_rmse:%r' % (np.min(test_rmse)),
      'min_mae:%r' % (test_mae[index]),
      'max_acc:%r' % (test_acc[index]),
      'r2:%r' % (test_r2[index]),
      'var:%r' % test_var[index])
