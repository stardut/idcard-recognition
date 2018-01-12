# -*- coding: utf-8 -*-

import os
import cv2
import random
from PIL import Image, ImageDraw, ImageFont

def get_char():
    char = set()
    with open('dict.txt', 'r', encoding='utf8') as f:
        while 1:
            text = f.readline()
            if len(text) < 2:
                break
            word = text.split(':')[0]
            char.add(word)
    return list(char)

def captcha_generator():
    
    set_chas = ['0123456789X']
    set_chas.append(get_char())
    nb_chas = [1]
    nb_image = 50000
    font_paths = ['../front/fangzheng.ttf', 'front/huawen.ttf']
    rotates = [False]

    dir_path = 'data/'
    bg_path = '../background/'
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    

    for i in range(nb_image):
        choose = random.randint(0, 1)
        is_cn = False
        if choose == 1:
            is_cn = True
        set_cha = set_chas[choose]
        font_path = font_paths[choose]
        rotate = random.choice(rotates)
        nb_cha = random.choice(nb_chas)

        height_im = 40

        label = 1
        # label = 0 if i < nb_image/2 else 1

        save_path = os.path.join(dir_path, str(label))
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # 1 是正样本
        if label == 1:
            nb_cha = random.randint(3, 6)
        weight_im = nb_cha * height_im

        if not is_cn:
            weight_im = int(weight_im * 0.5)
        else:
            weight_im = int(weight_im * 0.8)
        size_im = (weight_im, height_im)
        img = captcha_draw(size_im=size_im, nb_cha=nb_cha, set_cha=set_cha, bg_dir = bg_path,
            rotate=rotate, font=font_path, is_cn=is_cn)

        if label == 1:
            x1 = int(random.randint(0, 3) * 0.1 * weight_im)
            x2 = weight_im - x1
            img = img.crop((x1, 0, x2, height_im))
        else:
            x1 = int(random.randint(3, 4) * 0.1 * weight_im)
            x2 = int(random.randint(6, 8) * 0.1 * weight_im)
            y1 = int(random.randint(0, 5) * 0.1 * height_im)
            y2 = int(random.randint(6, 10) * 0.1 * height_im)
            img = img.crop((x1, y1, x2, y2))
        img_path = os.path.join(save_path, str(i) + '.jpg')
        print(img_path)
        img = img.convert('L')
        img.save(img_path)

def captcha_draw(size_im, nb_cha, set_cha, font=None, bg_dir='',
    rotate=False, img_num=0, img_now=0, is_cn=False):

    width_im, height_im = size_im
    text_color = 'black'   

    bg_path = os.path.join(bg_dir, random.choice(os.listdir(bg_dir)))
    bg = Image.open(bg_path)
    im = bg.resize((width_im, height_im))

    rate = random.randint(4, 8) * 0.1
    derx = int(height_im * (1.0 - rate - 0.05))
    dery = int(height_im * (1.0 - rate - 0.05))
    size_cha = int(height_im * rate)

    drawer = ImageDraw.Draw(bg)
    font = ImageFont.truetype(font, size_cha)

    for i in range(nb_cha):
        cha = random.choice(set_cha)
        im_cha = cha_draw(cha, text_color, font, rotate, size_cha)
        if is_cn:
            step = size_cha*i
        else:
            step = int(size_cha * 0.7 * i)
        im.paste(im_cha, (derx + step, dery), im_cha) # 字符左上角位置
    
    return im

def cha_draw(cha, text_color, font, rotate,size_cha):
    im = Image.new(mode='RGBA', size=(size_cha*2, size_cha*6))
    drawer = ImageDraw.Draw(im) 
    #text 内容，fill 颜色， font 字体（包括大小）
    drawer.text(xy=(0, 0), text=cha, fill=text_color, font=font) 
    if rotate:
        max_angle = 10 # to be tuned
        angle = random.randint(-max_angle, max_angle)
        im = im.rotate(angle, Image.BILINEAR, expand=1)
    im = im.crop(im.getbbox())
    return im

if __name__ == '__main__':
    captcha_generator()