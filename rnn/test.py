from data import Data
import cv2
import random
import numpy as np

data = Data()
for _ in range(30):
    imgs, labels = data.get_batch(1, random.randint(2, 30), 32)
    # # print(labels)
    # print(imgs[0])
    # cv2.imshow('test', imgs[0])
    # cv2.waitKey(0)
    cv2.imwrite('test.jpg', cv2.transpose(imgs[0]))