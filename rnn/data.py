# -*- coding:utf8 -*-
import os
import sys
import cv2
import random
import numpy as np
import tensorflow as tf
import words

from PIL import Image, ImageDraw, ImageFont

class Generator(object):
    """Generate image with word"""
    def __init__(self):
        super(Generator, self).__init__()
        self.id_char = dict([(idx, char) for idx, char in enumerate(words.chars)])
        self.char_id = dict([(char, idx) for idx, char in enumerate(words.chars)])

    def word_img(self, char, raw, img):
        col = random.randint(raw-5, raw)
        img = img.resize((col, raw)).convert("RGBA")
        res = img.copy()
        drawer = ImageDraw.Draw(img)
        
        font_size = random.randint(col-3, col)
        x = random.randint(0, col - font_size)
        y = random.randint(0, raw - font_size)
        font_path = os.path.join('font', random.choice(os.listdir('font')))
        font = ImageFont.truetype(font_path, font_size)
        fill = (random.randint(0, 200), random.randint(0, 200), random.randint(0, 200))
        drawer.text((x, y), text=char, fill=fill, font=font)

        angle = random.randint(-20, 20)
        img = img.rotate(angle, expand=0)
        res = Image.composite(img, res, img).convert("RGB")
        res = np.array(res)
        return res

    def get_background(self):
        b_path = 'background'
        imgs = os.listdir(b_path)
        img = Image.open(os.path.join(b_path, random.choice(imgs)))
        width, height = img.size[0], img.size[1]
        x = random.randint(0, width-50)
        y = random.randint(0, height-50)
        img = img.crop((x, y, x+40, y+40))
        return img

    def gen(self, batch_size, word_size):
        '''Generate batch_size images and word_size words in every image'''
        images = []
        labels = []
        for _ in range(batch_size):
            label = []
            ims = []
            cols = []
            raw = random.randint(20, 36)
            img = self.get_background()
            for i in range(word_size):
                idx = random.randint(0, len(self.char_id)-1)
                label.append(idx)                
                im = self.word_img(self.id2char(idx), raw, img)
                ims.append(im)
                cols.append(im.shape[1])
            cols = np.asarray(cols)
            img = np.zeros((raw, cols.sum(), 3), np.uint8)

            x = 0
            for idx, im in enumerate(ims):
                img[:, x:x+cols[idx], :] = im
                x += cols[idx]

            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            images.append(img)
            labels.append(np.asarray(label))
        return images, labels

    def char2id(self, char):
        return self.char_id[char]

    def id2char(self, idx):
        return self.id_char[idx]

class Data(object):
    """Data class for create training data and decode/encode data."""
    def __init__(self):
        self.generator = Generator()
        self.word_num = len(words.chars)

    def get_batch(self, batch_size=50, word_size=4, input_size=32):
        """Create a training batch of batch_size.

        Args:
            batch_size: a int number of batch size
        Return:
            images, lables, seq length
        """
        # imgs, labels = img_gen.captcha_generator(batch_size, self.word_dict)
        imgs, labels = self.generator.gen(batch_size, word_size)
        ims = []
        seq_len = []
        for im in imgs:
            im = np.asarray(im)
            raws, cols = im.shape
            im = cv2.resize(im, (input_size*word_size, input_size))
            im = np.transpose(im)
            ims.append(im)
        labels = self.sparse_tuple_from(labels)
        return np.asarray(ims), labels

    def sparse_tuple_from(self, sequences, dtype=np.int32):
        """Create a sparse representention of x.

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

    def decode_sparse_tensor(self, sparse_tensor):
        """Transform sparse to sequences ids."""
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
            text = ''.join(list(map(self.generator.id2char, ids)))
            result.append(text)
        return result

    def hit(self, text1, text2):
        """Calculate accuracy of predictive text and target text."""
        res = []
        for idx, words1 in enumerate(text1):
            res.append(words1 == text2[idx])
        return np.mean(np.asarray(res))