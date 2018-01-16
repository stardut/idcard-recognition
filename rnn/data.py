# -*- coding:utf8 -*-

import os
import numpy as np
import tensorflow as tf

sys.path.append('../genarate_data/recognition_data')
from set_dict import word_dict
import recognition_sample_gen as img_gen

class data(object):
    """docstring for data"""
    def __init__(self, img_shape):
        self.word_dict = word_dict()
        self.img_shape = img_shape

    def get_batch(self, batch_size):
        imgs, labels = img_gen(batch_size, self.word_dict)
        ims = []
        for im in imgs:
            im = np.resize(im, self.img_shape)
            ims.append(im)
        
            np.asarray(labels.split('-')) for label in labels


        return np.array(ims), 