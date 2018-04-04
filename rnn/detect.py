# -*- coding:utf-8 -*-
import os
import cv2
import sys
import numpy as np
import tensorflow as tf
from lstm import LSTM_CTC
from data import Data
from Levenshtein import *

import glob
import shutil
sys.path.append(os.getcwd())
sys.path.append('ctpn/')
from ctpn.lib.networks.factory import get_network
from ctpn.lib.fast_rcnn.config import cfg,cfg_from_file
from ctpn.lib.fast_rcnn.test import test_ctpn
from ctpn.lib.utils.timer import Timer
from ctpn.lib.text_connector.detectors import TextDetector
from ctpn.lib.text_connector.text_connect_cfg import Config as TextLineCfg

class Detect(object):
    """docstring for ctpn"""
    def __init__(self, sess):
        super(Detect, self).__init__()
        cfg_from_file('ctpn/ctpn/text.yml')

        # init session
        # config = tf.ConfigProto(allow_soft_placement=True)
        # sess = tf.Session(config=config)

        # sess = tf.Session()
        # load network
        net = get_network("VGGnet_test")
        # load model
        print(('Loading network {:s}... '.format("VGGnet_test")))
        saver = tf.train.Saver()

        ctpn_model_path = 'ctpn/checkpoints/VGGnet_fast_rcnn_iter_50000.ckpt'
        print('Restoring from {}...'.format(ctpn_model_path))
        saver.restore(sess, ctpn_model_path)
        print('Done\n')

        self.sess = sess
        self.net = net

    def resize_im(self, im, scale, max_scale=None):
        f=float(scale)/min(im.shape[0], im.shape[1])
        if max_scale!=None and f*max(im.shape[0], im.shape[1])>max_scale:
            f=float(max_scale)/max(im.shape[0], im.shape[1])
        return cv2.resize(im, None,None, fx=f, fy=f,interpolation=cv2.INTER_LINEAR), f


    def draw_boxes(self, img, image_name, boxes, scale):
        base_name = image_name.split('\\')[-1]
        res = []
        for box in boxes:
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue
            if box[8] >= 0.9:
                color = (0, 255, 0)
            elif box[8] >= 0.8:
                color = (255, 0, 0)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
            cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)

            min_x = min(int(box[0]/scale),int(box[2]/scale),int(box[4]/scale),int(box[6]/scale))
            min_y = min(int(box[1]/scale),int(box[3]/scale),int(box[5]/scale),int(box[7]/scale))
            max_x = max(int(box[0]/scale),int(box[2]/scale),int(box[4]/scale),int(box[6]/scale))
            max_y = max(int(box[1]/scale),int(box[3]/scale),int(box[5]/scale),int(box[7]/scale))

            line = ','.join([str(min_x),str(min_y),str(max_x),str(max_y)])+'\r\n'

            res.append([min_x, min_y, max_x, max_y])
        img = cv2.resize(img, None, None, fx=1.0/scale, fy=1.0/scale, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join("test/results", base_name), img)
        
        return res, img

    def ctpn(self, image_name):
        timer = Timer()
        timer.tic()

        img = cv2.imread(image_name)
        img, scale = self.resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
        scores, boxes = test_ctpn(self.sess, self.net, img)

        textdetector = TextDetector()
        boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])        
        timer.toc()
        print(('Detection took {:.3f}s for '
               '{:d} object proposals').format(timer.total_time, boxes.shape[0]))

        return self.draw_boxes(img, image_name, boxes, scale)
