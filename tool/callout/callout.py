import os
import cv2
import numpy as np
import shutil

data_path = 'data'
a_path = 'callout_img'
if not os.path.exists(a_path):
    os.mkdir(a_path)
img_save_path = 'img'
if not os.path.exists(img_save_path):
    os.mkdir(img_save_path)

cordi = []
def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_FLAG_LBUTTON:
        cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
        cordi.append(x)
        cordi.append(y)


with open('label', 'a') as f:
    for i in os.listdir(data_path):
        img = cv2.imread(os.path.join(data_path, i))
        print(i)
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', draw_circle)
        while 1:
            cv2.imshow('image', img)

            if cv2.waitKey(1)&0xFF == ord('r'):
                img = cv2.imread(os.path.join(data_path, i))
                cordi.clear()
            if cv2.waitKey(30)&0xFF == ord('n'):
                if len(cordi) != 8:
                    print('Error, please try again!!!')
                    cordi.clear()
                    img = cv2.imread(os.path.join(data_path, i))
                else:
                    info = '%s %d %d %d %d %d %d %d %d\n' % (i, cordi[0], cordi[1], cordi[2], 
                        cordi[3], cordi[4], cordi[5], cordi[6], cordi[7])
                    f.write(info)
                    cordi.clear()
                    cv2.imwrite(os.path.join(img_save_path, i), img)
                    shutil.move(os.path.join(data_path, i), os.path.join(a_path, i))
                    break
            if cv2.waitKey(1)&0xFF == ord('q'):
                exit()

        cv2.destroyAllWindows()
