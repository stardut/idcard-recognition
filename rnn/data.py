# -*- coding:utf8 -*-

import os
import sys
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
            im = np.resize(im, self.img_shape)
            im = np.transpose(im)
            ims.append(im)
        
        # labels = [np.asarray(label) for label in labels]
        print(labels)
        labels = self.sparse_tuple_from(labels)
        # labels = self.SimpleSparseTensorFrom(labels)
        print(labels)

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
            text = ''.join(list(map(word_dict.id2word, ids)))                
            result.append(text)

        return result


    def SimpleSparseTensorFrom(self, x):
        """Create a very simple SparseTensor with dimensions (batch, time).
        Args:
        x: a list of lists of type int
        Returns:
        x_ix and x_val, the indices and values of the SparseTensor<2>.
        """
        x_ix = []
        x_val = []
        for batch_i, batch in enumerate(x):
            for time, val in enumerate(batch):
                x_ix.append([batch_i, time])
                x_val.append(val)
        x_shape = [len(x), np.asarray(x_ix).max(0)[1]+1]
        x_ix = tf.constant(x_ix, tf.int64)
        x_val = tf.constant(x_val, tf.int32)
        x_shape = tf.constant(x_shape, tf.int64)

        return tf.SparseTensor(x_ix, x_val, x_shape)