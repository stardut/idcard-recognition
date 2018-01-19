# -*- coding:utf8 -*-

import time
import sys
import os
import numpy as np
import tensorflow as tf

from data import Data
from lstm import LSTM_CTC

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model_path = 'model'
if not os.path.exists(model_path):
    os.mkdir(model_path)

img_shape = (32, 256)
data = Data(img_shape)

num_layer = 2
num_units = 256
num_class = data.word_dict.word_num + 1 # 1: ctc_blank
keep_prob = 0.5
input_size = img_shape[0]
time_step = img_shape[1]

learn_rate = 0.001
batch_size = 128
step = 10000 * 100


model = LSTM_CTC(num_layer=num_layer,
                 num_units=num_units,
                 num_class=num_class,
                 input_size=input_size,
                 batch_size=batch_size,
                 learn_rate=learn_rate,
                 time_step=time_step)

model.build()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
    start = time.time()
    for i in range(step):
        inputs, labels = data.get_batch(batch_size)
        seq_len = np.ones(batch_size) * img_shape[1]

        feed = {
            model.X : inputs,
            model.labels : labels,
            model.seq_len : seq_len,
            model.keep_prob: keep_prob
        }
        decoded, loss, _ = sess.run([model.decoded, model.loss, model.train_op], feed_dict=feed)

        if step % 100 == 0:
            pre = data.decode_sparse_tensor(decoded[0])
            ori = data.decode_sparse_tensor(labels)
            acc = data.hit(pre, ori)
            t = (time.time() - start()) / 100.0
            print('step: {}, accuracy: {:.4f}, loss: {:.6f}, time cost per step: {}'.format(i, acc, loss, t))

        if step % 10000 == 0:
            checkpoint_path = os.path.join(model_path, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)