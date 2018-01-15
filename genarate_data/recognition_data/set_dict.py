# -*- coding: utf8 -*-

import os
from six.moves import cPickle


class word_dict(object):
    """set Chinese-id dict"""
    def __init__(self):
        super(word_dict, self).__init__()
        self.vocab_file = 'vocab.pkl'
        self.init()

    def init(self):
        if os.path.exists(self.vocab_file):
            with open(self.vocab_file, 'rb') as f:
                print('load vocab file from {}'.format(self.vocab_file))
                self.words = cPickle.load(f)
        else:
            self.set_vocab()
        self.word_num = len(self.words)
        print('word number: {}'.format(self.word_num))
        self.word_id = {word : idx for idx, word in enumerate(self.words)}
        self.id_word = dict(enumerate(self.words))

    def set_vocab(self):
        with open('word1.txt', 'r', encoding='utf8') as f:
            word = set(f.read())
            self.words = list(word)
            with open(self.vocab_file, 'wb') as f1:
                cPickle.dump(self.words, f1)
                print('save vocab file to {}'.format(self.vocab_file))

    def word2id(self, word):
        return self.word_id[word]

    def id2word(self, idx):
        return self.id_word[idx]