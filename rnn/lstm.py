# -*- coding:utf8 -*-

import os
import time
import tensorflow as tf
import numpy as np

class LSTM_CTC(object):
    """docstring for LSTM"""
    def __init__(self, num_layer, num_units, keep_prob, 
            num_class, labels, input_size, batch_size):
        self.num_layer = num_layer
        self.num_units = num_units
        self.num_class = num_class
        self.keep_prob = keep_prob
        self.labels    = labels
        self.input_size = input_size
        self.batch_size = batch_size

    def build(self):
        self.global_step = tf.Variable(0, trainable=False)
        self.X = tf.placeholder(dtype=tf.float32, [None, None, self.input_size])
        self.labels = tf.sparse_placeholder(tf.int32)
        self.seq_len = tf.placeholder(tf.int32, [None])

        mlstm_cell = tf.nn.rnn_cell.MultiRNNCell([self.unit() for i in range(self.num_layer)])
        self.init_state = mlstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        outputs, state = tf.nn.dynamic_rnn(cell=mlstm_cell, 
                                           inputs=self.X,
                                           sequence_length=self.seq_len,
                                           initial_state=self.init_state,
                                           time_major=False)

        h_out = outputs[:, -1, :]
        W = tf.Variable(tf.truncated_normal([self.num_units, self.num_class], stddev=0.1), 
            dtype=tf.float32)
        b = tf.Variable(tf.constant(0.1, shape=[self.num_class], dtype=tf.float32))
        
        logits = tf.nn.softmax(tf.matmul(h_out, W) + b)
        logits = tf.reshape(logits, [self.batch_size, -1, self.num_class])
        logits = tf.transpose(logits, (1, 0, 2))

        self.loss = tf.reduce_mean(
            tf.nn.ctc_loss(labels=self.labels, inputs=logits, sequence_length=self.seq_len))
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss, self.global_step)

        self.decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)    
        self.acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), self.labels))


    def unit(self):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.num_units)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
        return lstm_cell
        