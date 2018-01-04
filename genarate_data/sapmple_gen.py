# -*- coding: utf-8 -*-

import os
import cv2
import random
from PIL import Image, ImageDraw, ImageFont


def captcha_generator():
    size_im = (200, 200)
    set_chas = ['0123456789X', '天好']
    nb_chas = [1]
    nb_image = 500
    font_paths = ['front/fangzheng.ttf', 'front/huawen.ttf']
    rotates = [True]

    dir_path = 'data/'
    for i in range(nb_image):
        choose = random.randint(0, 1)
        set_cha = random.choice(set_chas[choose])
        font_path = random.choice(font_paths[choose])
        rotate = random.choice(rotates)
        nb_cha = random.choice(nb_chas)

        label = random.randint(0, 1)
        captcha_draw(size_im=size_im, nb_cha=nb_cha, set_cha=set_cha, bg_path = bg_path,
            rotate=rotate, dir_path=os.path.join(dir_path, str(label)), fonts=font_path)


def captcha_draw(size_im, nb_cha, set_cha, fonts=None, bg_path
    rotate=False, dir_path='', img_num=0, img_now=0):
    """
        字体大小 目前长宽认为一致！
        所有字大小一致
        fonts 中分中文和数字
    """
    rate_cha = 0.9 # rate to be tuned
    width_im, height_im = size_im
    width_cha = int(width_im / max(nb_cha-overlap, 3)) # 字符区域宽度
    height_cha = height_im*1.2# 字符区域高度
    bg_color = 'white'
    text_color = 'black'
    derx = 0
    dery = 0

    im = Image.new(mode='GRAY', size=size_im, color=bg_color) # color 背景颜色，size 图片大小
    Image.open()

    drawer = ImageDraw.Draw(im)
    contents = []
    for i in range(nb_cha):
        if rd_text_pos:
            derx = random.randint(0, max(width_cha-size_cha-5, 0))
            dery = random.randint(0, max(height_cha-size_cha-5, 0))

        cha = random.choice(set_cha)
        font = ImageFont.truetype(fonts['eng'], size_cha)
        contents.append(cha) 
        im_cha = cha_draw(cha, text_color, font, rotate, size_cha)
        im.paste(im_cha, (int(max(i-overlap, 0)*width_cha)+derx + 2, dery + 3), im_cha) # 字符左上角位置
    
    if os.path.exists(dir_path) == False: # 如果文件夹不存在，则创建对应的文件夹
        os.mkdir(dir_path )

    img_name = str(img_now) + '_' + ''.join(contents) + '.jpg'
    img_path = dir_path + img_name
    print img_path, str(img_now) +  '/' + str(img_num)
    im.save(img_path)

def cha_draw(cha, text_color, font, rotate,size_cha):
    im = Image.new(mode='RGBA', size=(size_cha*2, size_cha*2))
    drawer = ImageDraw.Draw(im) 
    #text 内容，fill 颜色， font 字体（包括大小）
    drawer.text(xy=(0, 0), text=cha, fill=text_color, font=font) 
    if rotate:
        max_angle = 20 # to be tuned
        angle = random.randint(-max_angle, max_angle)
        im = im.rotate(angle, Image.BILINEAR, expand=1)
    im = im.crop(im.getbbox())
    return im