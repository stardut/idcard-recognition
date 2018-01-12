# -*- coding:utf-8 -*-

import re
import os

idx = 0
wd = set()
with open('现代汉语词典（第五版）全文.txt', 'r', encoding='utf8') as f:        
    while 1:
        text = f.readline()
        if len(text) < 1:
            break
        start = re.search('【', text)
        end = re.search('】', text)
        # print(start, end)
        if start == None or end == None:
            continue
        start = start.span()[1]
        end = end.span()[0]
        words = text[start:end].replace(' ', '')
        # print(words)
        # break
        for i in words:
            wd.add(i)
        
    

with open('dict.txt', 'w', encoding='utf8') as f:
    for i in wd:
        f.write('{}:{}\n'.format(i, idx))
        idx += 1