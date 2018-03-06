# -*- coding:utf-8 -*-

import os
import cv2
import numpy as np
import tensorflow as tf
from lstm import LSTM_CTC
from data import Data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Model(object):
    """docstring for model"""
    def __init__(self):
        super(Model, self).__init__()
        self.data = Data()
        num_layer = 2
        num_units = 512
        num_class = self.data.word_num + 1
        input_size = 32
        batch_size = 1
        model = LSTM_CTC(num_layer=num_layer,
                         num_units=num_units,
                         num_class=num_class,
                         input_size=input_size,
                         batch_size=batch_size)

        self.decoded, _ = tf.nn.ctc_beam_search_decoder(model.logits, model.seq_len//8, merge_repeated=False)
        sess = tf.Session()
        saver = tf.train.Saver(tf.global_variables())        
        saver.restore(sess, 'model/model')
        self.sess = sess
        self.model = model

    def predict(self, imgs):
        inputs = []
        seq_len = []
        for im in imgs:
            raw, col = im.shape
            x = int(col/raw*32) - (int(col/raw*32) % 8)
            im = cv2.resize(im, (x, 32))            
            im = cv2.transpose(im)
            
            inputs.append(im/255)
            seq_len.append(x)

        feed = {
            self.model.X : inputs,
            self.model.seq_len : seq_len,
            self.model.keep_prob: 1.0,
            self.model.is_train: False
        }
        decode = self.sess.run(self.decoded, feed_dict=feed)
        for d in decode:
            pre = self.data.decode_sparse_tensor(d)
            print(pre)
            cv2.imshow('test', np.transpose(inputs[0]))
            cv2.waitKey(0)

model = Model()

img = cv2.imread('test.png')
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

imgs = []
imgs.append(img)
model.predict(imgs)