# -*- coding: utf-8 -*-

import os
import cv2
import random
from PIL import Image, ImageDraw, ImageFont
from set_dict import word_dict


def captcha_generator(nb_image, word):
    font_paths = ['../front/fangzheng.ttf', 'front/huawen.ttf']
    rotates = [False]

    dir_path = 'train_data/'
    bg_path = '../background/'
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)    

    for i in range(nb_image):
        font_path = font_paths[0]
        rotate = random.choice(rotates)
        nb_cha = random.randint(1, 26)
        height_im = 40

        weight_im = nb_cha * height_im
        weight_im = int(weight_im * 0.7)
        size_im = (weight_im, height_im)
        img, label = captcha_draw(word=word, 
                           size_im=size_im, 
                           nb_cha=nb_cha, 
                           bg_dir=bg_path, 
                           font=font_path)
        imname = '{:0>6d}_{}.jpg'.format(i, label)
        img_path = os.path.join(dir_path, imname)
        print(img_path)
        img = img.convert('L')
        img.save(img_path)

def captcha_draw(word, size_im, nb_cha, font=None, bg_dir='', rotate=False):
    width_im, height_im = size_im
    text_color = 'black'

    bg_path = os.path.join(bg_dir, random.choice(os.listdir(bg_dir)))
    bg = Image.open(bg_path)
    im = bg.resize((width_im, height_im))

    rate = random.randint(2, 3) * 0.1
    derx = int(height_im * 0.1)
    dery = int(height_im * 0.2)
    size_cha = int(height_im * 0.6)

    drawer = ImageDraw.Draw(bg)
    font = ImageFont.truetype(font, size_cha)
    tmp = 10
    set_char = '0123456789X'
    if nb_cha > 10:        
        tmp = random.randint(0, 10)

    label = ''
    for i in range(nb_cha):
        cha_id = random.randint(0, word.word_num-1)
        cha = word.id2word(cha_id)
        # 生成全是数字和X的身份证号码
        if tmp < 2:
            cha = random.choice(set_char)
            cha_id = word.word2id(cha)
        label += str(cha_id)
        if i < nb_cha-1:
            label += '-'    
        
        im_cha = cha_draw(cha, text_color, font, rotate, size_cha)
        step = int(size_cha * i)
        im.paste(im_cha, (derx + step, dery), im_cha) # 字符左上角位置
    
    return im, label


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
    num = 50
    word = word_dict()
    captcha_generator(num, word)