# -*- coding:utf-8 -*-

# opencv 3.2.0
import cv2
import numpy as np

img = cv2.imread('test.jpg')
mser = cv2.MSER_create()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
regions, boxes = mser.detectRegions(gray)

for box in boxes:
    x, y, w, h = box
    cv2.rectangle(img, (x,y),(x+w, y+h), (255, 0, 0), 1)

cv2.imshow('test', img)
cv2.waitKey(0)