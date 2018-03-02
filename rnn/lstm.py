# -*- coding:utf8 -*-
import os
import time
import tensorflow as tf
import numpy as np

class LSTM_CTC(object):
    """LSTM and CTC network."""
    def __init__(self, num_layer, num_units, num_class, input_size, batch_size):
        self.num_layer = num_layer
        self.num_units = num_units
        self.num_class = num_class
        self.input_size = input_size
        self.batch_size = batch_size
        self.build()

    def build(self):
        """Build network."""
        self.global_step = tf.Variable(0, trainable=False)
        self.learn_rate = tf.placeholder(tf.float32, name='learn_rate')
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, None, self.input_size], name='input')
        self.labels = tf.sparse_placeholder(tf.int32, name='label')
        self.seq_len = tf.placeholder(tf.int32, [None], name='sequence_len')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.is_train = tf.placeholder(tf.bool)

        x = tf.reshape(self.X, [-1, self.seq_len[0], self.input_size, 1], name='inputs')
        w1 = self.weight_variable([3, 3, 1, 32])
        b1 = self.bias_variable([32])
        conv1 = tf.nn.relu(self.batch_norm(self.conv2d(x, w1) + b1, 32, self.is_train))

        w2 = self.weight_variable([3, 3, 32, 64])
        b2 = self.bias_variable([64])
        conv2 = tf.nn.relu(self.batch_norm(self.conv2d(conv1, w2) + b2, 64, self.is_train))
        pool1 = self.max_pool(conv2)

        w3 = self.weight_variable([3, 3, 64, 96])
        b3 = self.bias_variable([96])
        conv3 = tf.nn.relu(self.batch_norm(self.conv2d(pool1, w3) + b3, 96, self.is_train))

        w4 = self.weight_variable([3, 3, 96, 128])
        b4 = self.bias_variable([128])
        conv4 = tf.nn.relu(self.batch_norm(self.conv2d(conv3, w4) + b4, 128, self.is_train))
        pool2 = self.max_pool(conv4)

        w5 = self.weight_variable([3, 3, 128, 256])
        b5 = self.bias_variable([256])
        conv5 = tf.nn.relu(self.batch_norm(self.conv2d(pool2, w5) + b5, 256, self.is_train))
        pool3 = self.max_pool(conv5)

        w6 =self.weight_variable([3, 3, 256, 256])
        b6 = self.bias_variable([256])
        conv6 = tf.nn.relu(self.batch_norm(self.conv2d(pool3, w6) + b6, 256, self.is_train))

        cnn_out = tf.reshape(conv6, [self.batch_size, -1])

        feature = tf.nn.dropout(cnn_out, self.keep_prob)
        feature = tf.reshape(feature, [self.batch_size, self.seq_len[0]//8, 4*256])
        self.feature = feature
        
        mlstm_cell = tf.nn.rnn_cell.MultiRNNCell([self.unit() for i in range(self.num_layer)])
        init_state = mlstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        outputs, state = tf.nn.dynamic_rnn(cell=mlstm_cell,
                                           inputs=feature,
                                           sequence_length=self.seq_len//8,
                                           initial_state=init_state,
                                           time_major=False)

        h_state = tf.reshape(outputs, [-1, self.num_units])
        W = tf.Variable(tf.truncated_normal([self.num_units, self.num_class], stddev=0.1), 
            dtype=tf.float32, name='lstm_w')
        b = tf.Variable(tf.constant(0., shape=[self.num_class], dtype=tf.float32), name='lstm_b')

        logits = tf.matmul(h_state, W) + b
        logits = tf.reshape(logits, [self.batch_size, -1, self.num_class])
        self.logits = tf.transpose(logits, (1, 0, 2))
        
        self.loss = tf.nn.ctc_loss(labels=self.labels, inputs=self.logits, sequence_length=self.seq_len//8)
        self.cost = tf.reduce_mean(self.loss)
        self.train_op = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss, self.global_step)

    def unit(self):
        # lstm_cell = tf.nn.rnn_cell.GRUCell(num_units=self.num_units)
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.num_units)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
        return lstm_cell

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.5)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W, padding='SAME'):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)

    def max_pool(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def batch_norm(self, x, n_out, phase_train):
        '''Batch normalization on convolutional maps.

        Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
        Args:
            x:           Tensor, 4D BHWD input maps
            n_out:       integer, depth of input maps
            phase_train: boolean tf.Varialbe, true indicates training phase
            scope:       string, variable scope
        Return:
            normed:      batch-normalized maps
        '''
        with tf.variable_scope('bn'):
            beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                         name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                          name='gamma', trainable=True)
            batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(phase_train,
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
        return normed