# -*- coding: utf-8 -*-
import tensorflow as tf
from utils import calculate_laplacian
from keras import layers


def Model(_inputs, gru_units, adj, num_nodes, pre_len):
    cell = TGCN_Cell(gru_units, adj, num_nodes=num_nodes)
    last_output = layers.RNN(cell, unroll=True)(_inputs)
    last_output = tf.reshape(last_output, shape=[-1, gru_units])
    output = layers.Dense(pre_len)(last_output)
    output = tf.reshape(output, shape=[-1, num_nodes])
    return output


class TGCN_Cell(tf.keras.layers.AbstractRNNCell):
    """Temporal Graph Convolutional Network """

    def __init__(self, num_units, adj, num_nodes):
        super(TGCN_Cell, self).__init__()
        self._nodes = num_nodes
        self._units = num_units
        self._adj = calculate_laplacian(adj)

    @property
    def state_size(self):
        return self._nodes * self._units

    def __call__(self, inputs, state):
        with tf.name_scope("TGCN_Cell"):
            value = tf.nn.sigmoid(self._gc(inputs, state[0], 2 * self._units))
            r_t, u_t = tf.split(value=value, num_or_size_splits=2, axis=1)
            r_state = r_t * state[0]
            c_t = tf.nn.tanh(self._gc(inputs, r_state, self._units))
            h_t = u_t * state[0] + (1 - u_t) * c_t

        return h_t, h_t

    def _gc(self, inputs, state, output_size):
        with tf.name_scope("GC"):
            inputs = tf.expand_dims(inputs, 2)
            state = tf.reshape(state, (-1, self._nodes, self._units))
            x_s = tf.concat([inputs, state], axis=2)
            input_size = x_s.get_shape()[2].value
            x0 = tf.reshape(x_s, shape=[self._nodes, -1])
            x1 = tf.sparse.sparse_dense_matmul(self._adj, x0)
            x = tf.reshape(x1, shape=[-1, input_size])
            x = layers.Dense(self._nodes * output_size, activation="sigmoid")(x)
            x = tf.reshape(x, shape=[-1, self._nodes * output_size])
        return x
