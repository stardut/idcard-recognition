import os
import numpy as np
import cv2
import random

dir_path = 'idcard/'

files = os.listdir(dir_path)
num = 2000
name = 'bg_'
save_dir = 'data/0/'
idx = 0
for file in files:
    img_path = os.path.join(dir_path, file)
    # print(img_path)
    img = cv2.imread(img_path, 0)
    if img is None:
        continue
    # print(img.shape)
    img = cv2.resize(img, (450, 300))
    img = img[0:210, 280:]
    # cv2.imshow('test', img)
    # cv2.waitKey(0)
    
    for i in range(num):
        width = random.randint(5, 50)
        height = random.randint(5, 50)

        x1 = random.randint(0, 170)
        y1 = random.randint(0, 210)

        if x1+width > 170 or y1+height > 210:
            continue

        img_name = os.path.join(save_dir, name + str(idx) + '.jpg')
        print(img_name)
        cv2.imwrite(img_name, img[y1:y1+height, x1:x1+width])
        idx += 1
