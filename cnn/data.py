# -*- coding:utf8 -*-

import os
import numpy as np
import cv2
import random

class data():
    """docstring for data"""
    def __init__(self, data_dir, train=False):
        
        self.pointer = 0
        img_width = 56
        img_height = 56
        self.imgs = []        
        save_path = 'train_data.npy' if train else 'val_data.npy'

        if os.path.exists(save_path):
            print('Data loading from ' + save_path)
            self.imgs = np.load(save_path)
        else:
            print('Data loading from ' + data_dir)
            for folder in os.listdir(data_dir):
                label = int(folder)
                img_dir = os.path.join(data_dir, folder)
                
                for file in os.listdir(img_dir):
                    img_path = os.path.join(img_dir, file)
                    img = cv2.imread(img_path, 0)
                    img = cv2.resize(img, (img_height, img_width))
                    img = np.reshape(img, img_height*img_width)

                    data = []
                    data.append(img)
                    data.append(label)
                    self.imgs.append(data)

            random.shuffle(self.imgs)
            np.save(save_path, np.array(self.imgs))

        print('Done.')


    def get_data(self, batch_size=0):
        if batch_size != 0:
            epoch = len(self.imgs) // batch_size
            self.imgs = self.imgs[: epoch*batch_size]        
            if self.pointer == epoch*batch_size - 1:
                self.pointer = 0

            x = []
            y = []
            for i in range(batch_size):
                data = self.imgs[self.pointer*batch_size + i][0]
                label = self.imgs[self.pointer*batch_size + i][1]
                lab = np.zeros(2)
                lab[label] = 1
                x.append(data)
                y.append(lab)

            self.pointer += batch_size
            return np.array(x), np.array(y)

        else:
            batch_size = len(self.imgs)
            x = []
            y = []
            for i in range(batch_size):
                data = self.imgs[i][0]
                label = self.imgs[i][1]
                lab = np.zeros(2)
                lab[label] = 1
                x.append(data)
                y.append(lab)
            return np.array(x), np.array(y)