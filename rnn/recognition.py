# -*- coding:utf-8 -*-

import os
import cv2
import numpy as np
import tensorflow as tf
from lstm import LSTM_CTC
from data import Data
from Levenshtein import *

class Recognizer(object):
    """docstring for Recognizer"""
    def __init__(self, num_layer, num_units, input_size, batch_size, sess):
        super(Recognizer, self).__init__()
        self.data = Data()
        num_class = self.data.word_num + 1
        model = LSTM_CTC(num_layer=num_layer,
                         num_units=num_units,
                         num_class=num_class,
                         input_size=input_size,
                         batch_size=batch_size)

        self.decoded, _ = tf.nn.ctc_beam_search_decoder(model.logits, 
            model.seq_len//8, merge_repeated=False)
        # sess = tf.Session()
        print('Restore recognition model.')
        saver = tf.train.Saver(tf.global_variables())
        model_file = tf.train.latest_checkpoint('model')
        model_file = 'model/model'
        saver.restore(sess, model_file)
        print('Done.\n')
        self.sess = sess
        self.model = model

    def predict(self, images):
        imgs = self.img_process(images)
        imgs, seq_len = self.data.scale(imgs, 32)

        inputs = []
        for img in imgs:
            img = np.transpose(img)
            inputs.append(img/255)

        feed = {
            self.model.X : inputs,
            self.model.seq_len : seq_len,
            self.model.keep_prob: 1.0,
            self.model.is_train: False
        }
        decode = self.sess.run(self.decoded, feed_dict=feed)
        pre = self.data.decode_sparse_tensor(decode[0])        
        return pre

    # 预处理
    def img_process(self, images):
        imgs = []
        for img in images:
            if len(img.shape) > 2:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = self.equ(img)
            img = self.cut(img)            
            imgs.append(img)
        return imgs

    # 均衡化
    def equ(self, image):
        lut = np.zeros(256, dtype=image.dtype)
        hist= cv2.calcHist([image], [0], None, [256], [0.0,255.0])
        minBinNo, maxBinNo = 0, 255  

        for binNo, binValue in enumerate(hist):  
            if binValue != 0:  
                minBinNo = binNo  
                break 

        for binNo, binValue in enumerate(reversed(hist)):  
            if binValue != 0:  
                maxBinNo = 255-binNo  
                break

        for i, v in enumerate(lut):
            if i < minBinNo:  
                lut[i] = 0  
            elif i > maxBinNo:  
                lut[i] = 255  
            else:  
                lut[i] = int(255.0*(i-minBinNo)/(maxBinNo-minBinNo)+0.5)  
 
        result = cv2.LUT(image, lut)
        return result

    # 裁掉边缘多余的空白
    def cut(self, img):
        col = np.mean(img, axis=0)
        row = np.mean(img, axis=1)

        col_threshold = 220
        row_threshold = 210
        col_edge_len = 5
        row_edge_len = 3
        left, right, top, buttom = 0, len(col), 0, len(row)
        
        for idx, i in enumerate(col):
            if i < col_threshold:
                left = max(idx - col_edge_len, 0)
                break

        for idx, i in enumerate(row):
            if i < row_threshold:
                top = max(idx - row_edge_len, 0)
                break

        for idx in reversed(range(len(col))):
            if col[idx] < col_threshold:
                right = min(idx + col_edge_len, len(col))
                break

        for idx in reversed(range(len(row))):
            if row[idx] < row_threshold:
                buttom = min(idx + row_edge_len, len(row))
                break
        return img[top:buttom, left:right]


'''
model = Model(num_layer, num_units, input_size, batch_size)
root = 'test/image/'
lable = 'test/label.txt'

num = 0
hit = 0
err = 0
word_num = 0

with open(lable, 'r') as f:
    data = f.readlines()
    labels = []
    img_paths = []
    for i in data:
        s = i.split(' ')
        img_path = os.path.join(root, s[0] + '.jpg')
        label = s[1].replace('\n', '')
        label = label.replace(' ', '')
        labels.append(label)
        img_paths.append(img_path.replace('\ufeff', ''))
    epoch = len(labels) // batch_size

    for i in range(epoch):
        res = pre(model, img_paths[i*batch_size : (i+1)*batch_size])
        res = list(map(lambda x: x.replace(' ', ''), res))
        num += batch_size
        
        for m in range(batch_size):
            if labels[i*batch_size + m] == res[m]:
                hit += 1

            err += distance(labels[i*batch_size+m], res[m])
            word_num += len(labels[i*batch_size+m])

            print('ori: %s\npre: %s  %d\n' % (labels[i*batch_size+m], res[m], num))

    print('accuracy: %.3f, word error: %.3f' % (hit/num, err/word_num))

model.sess.close()
'''
