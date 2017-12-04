# -*- coding:utf-8 -*-

# opencv 3.2.0
import cv2
import numpy as np
import nms

img = cv2.imread('test2.jpg')
mser = cv2.MSER_create(_max_area=300)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
regions, boxes = mser.detectRegions(gray)
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

keep = []
for box in hulls:
    x, y, w, h = cv2.boundingRect(box)
    # cv2.rectangle(img, (x,y),(x+w, y+h), (255, 0, 0), 1)
    keep.append([x, y, x + w, y + h])

keep2=np.array(keep)
pick = nms.nms(keep2, 0.1)

for (startX, startY, endX, endY) in pick:
    cv2.rectangle(img, (startX, startY), (endX, endY), (255, 0, 0), 1)

cv2.imshow('test', img)
cv2.waitKey(0)