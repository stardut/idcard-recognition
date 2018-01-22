# -*- coding:utf8 -*-

import os
import sys
import cv2
import numpy as np
import tensorflow as tf

sys.path.append('../genarate_data/recognition_data/')
from set_dict import word_dict
import recognition_sample_gen as img_gen

class Data(object):
    """docstring for data"""
    def __init__(self, img_shape):
        self.word_dict = word_dict()
        self.img_shape = img_shape

    def get_batch(self, batch_size=50):
        imgs, labels = img_gen.captcha_generator(batch_size, self.word_dict)
        ims = []
        for im in imgs:
            im = cv2.resize(im, (self.shape[1], self.shape[0]))
            im = np.transpose(im)
            ims.append(im)
        labels = self.sparse_tuple_from(labels)
        return np.asarray(ims), labels


    #转化一个序列列表为稀疏矩阵    
    def sparse_tuple_from(self, sequences, dtype=np.int32):
        """
        Create a sparse representention of x.
        Args:
            sequences: a list of lists of type dtype where each element is a sequence
        Returns:
            A tuple with (indices, values, shape)
        """
        indices = []
        values = []        
        for n, seq in enumerate(sequences):
            indices.extend(zip([n] * len(seq), range(len(seq))))
            values.extend(seq)
     
        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(values, dtype=dtype)
        shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)     
        return indices, values, shape


    #转化稀疏矩阵到序列
    def decode_sparse_tensor(self, sparse_tensor):
        """
        transform sparse to sequences ids
        """
        decoded_indexes = list()
        current_i = 0
        current_seq = []
        for offset, i_and_index in enumerate(sparse_tensor[0]):
            i = i_and_index[0]
            if i != current_i:
                decoded_indexes.append(current_seq)
                current_i = i
                current_seq = list()
            current_seq.append(offset)
        decoded_indexes.append(current_seq)

        result = []
        for index in decoded_indexes:
            ids = [sparse_tensor[1][m] for m in index]
            text = ''.join(list(map(self.word_dict.id2word, ids)))
            result.append(text)
        return result


    def hit(self, text1, text2):
        res = []
        for idx, words in enumerate(text1):
            res.append(words == text2[idx])
        accurary = np.mean(np.asarray(res))
        return accurary


        