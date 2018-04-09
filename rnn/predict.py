# -*- coding:utf-8 -*-

import os
import cv2
import time
import numpy as np
import tensorflow as tf
from detect import Detect
from recognition import Recognizer

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Recognizer
num_layer = 2
num_units = 512
input_size = 32
batch_size = 10

g1 = tf.Graph()
g2 = tf.Graph()
sess1 = tf.Session(graph=g1)
sess2 = tf.Session(graph=g2)

with g1.as_default():
    recognizer = Recognizer(num_layer, num_units, input_size, batch_size, sess1)

with g2.as_default():
    detector = Detect(sess2)

def run(img_path):
    boxs, img_box = detector.ctpn(img_path)
    img = cv2.imread(img_path)

    boxs = sorted(boxs, key=lambda x: x[1])

    imgs = []
    for b in boxs:
        x1, y1, x2, y2 = b
        imgs.append(img[y1:y2, x1:x2])

    img_num = len(imgs)
    while len(imgs) < batch_size:
        imgs.append(img[y1:y2, x1:x2])

    print('image number: %d\n' % (img_num))

    res = recognizer.predict(imgs[:batch_size])
    for idx in range(min(img_num, batch_size)):
        print(res[idx])

    y, x, c = img_box.shape
    y1 = 400
    x1 = int(y1*x/y)

    img_box = cv2.resize(img_box, (x1, y1))
    cv2.imshow('test', img_box)
    cv2.waitKey(0)

# path = 'test/img/'
# for im in os.listdir(path):
#     run(os.path.join(path, im))

run('test/img/test7.png')