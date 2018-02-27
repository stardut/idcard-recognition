# -*- coding:utf8 -*-
import time
import sys
import os
import cv2
import random
import numpy as np
import tensorflow as tf
from data import Data
from lstm import LSTM_CTC

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_path = 'model'
if not os.path.exists(model_path):
    os.mkdir(model_path)

data = Data()
input_size = 32

word_size = 8 # words in every image
num_layer = 2
num_units = 512
num_class = data.word_num + 1 # 1: ctc_blank
keep_prob = 0.5
learn_rate = 0.001
batch_size = 128
step = 10000 * 100

model = LSTM_CTC(num_layer=num_layer,
                 num_units=num_units,
                 num_class=num_class,
                 input_size=input_size,
                 batch_size=batch_size)
model.build()
decoded, log_pro = tf.nn.ctc_beam_search_decoder(model.logits, model.seq_len//8, merge_repeated=False)
err = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), model.labels))
init = tf.global_variables_initializer()

print('\n========')
print('classes number:', num_class)
print('========\n')

with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    start = time.time()
    for i in range(step):
        word_size = random.randint(4, 10)
        inputs, labels = data.get_batch(batch_size=batch_size, 
                                        word_size=word_size, 
                                        input_size=input_size)

        seq_len = np.ones(batch_size) * (input_size*word_size)
        feed = {
            model.X : inputs,
            model.labels : labels,
            model.seq_len : seq_len,
            model.keep_prob: keep_prob,
            model.learn_rate: learn_rate,
            model.is_train: True
        }

        if (i+1) % 100 == 0:
            feed[model.keep_prob] = 1.0
            feed[model.is_train] = False
            decode, word_error, cost = sess.run([decoded, err, model.cost], feed_dict=feed)
            pre = data.decode_sparse_tensor(decode[0])
            ori = data.decode_sparse_tensor(labels)
            acc = data.hit(pre, ori)
            speed = i * batch_size / (time.time() - start)
            print(('step: %d, accuracy: %.4f, word error: %.4f loss: %.6f, speed: %.2f imgs/s, lr: %.6f') % \
                (i+1, acc, word_error, cost, speed, learn_rate))
            print('origin : ' + ori[0])
            print('predict: ' + pre[0])

        sess.run(model.train_op, feed_dict=feed)

        if (i+1) % 5000 == 0:
            learn_rate = max(0.98 * learn_rate, 0.00001)
            name = 'model_acc_%.1f_loss_%.4f.ckpt'
            checkpoint_path = os.path.join(model_path, name)
            saver.save(sess, checkpoint_path, global_step=i+1)
            print('save model in step: {}'.format(i+1))