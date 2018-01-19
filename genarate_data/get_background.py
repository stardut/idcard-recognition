import os
import cv2
import random

save_dir = 'background/'
if not os.path.exists(save_dir)
    os.mkdir(save_dir)
num = 0
imgs = ['fang.jpg', 'zheng.jpg']
for i in range(2000):
    img = cv2.imread(random.choice(imgs))
    raws, cols, channel = img.shape
    width = random.randint(20, 100)
    height = random.randint(20, 100)
    x1 = random.randint(0, cols-10 - width)
    y1 = random.randint(0, raws-10 - height)

    img_name = save_dir + str(num) + '.jpg'
    cv2.imwrite(img_name, img[y1:y1+height, x1:x1+width])
    print(img_name)
    num += 1
