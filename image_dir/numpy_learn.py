# -*- coding: utf-8 -*-

"""
Copyright () 2018

All rights reserved

FILE: numpy_learn.py
AUTHOR: tianyuningmou
DATE CREATED:  @Time : 2018/3/15 下午3:36

DESCRIPTION:  .

VERSION: : #1 
CHANGED By: : tianyuningmou
CHANGE:  : 
MODIFIED: : @Time : 2018/3/15 下午3:36
"""

from PIL import Image
from matplotlib.pylab import *
from image_dir import imtool

image1 = np.array(Image.open('fbb.jpeg'))
print(image1.shape, image1.dtype)

image1 = np.array(Image.open('fbb.jpeg').convert('L'), 'f')
print(image1.shape, image1.dtype)

im1 = np.array(Image.open('fbb.jpeg').convert('L'))
# 对图像进行反相处理
im2 = 255 - im1
# 将图像像素值变换到100-200之间
im3 = (100.0/255) * im1 + 100
# 对图像像素值求平方后得到的图像
im4 = 255.0 * (im1/255.0)**2

image2 = Image.fromarray(im2)
image2.save('fbb_2.jpeg')

image3 = Image.fromarray(np.uint8(im3))
image3.save('fbb_3.jpeg')

image4 = Image.fromarray(np.uint8(im4))
image4.save('fbb_4.jpeg')

im = np.array(Image.open('fbb.jpeg').convert('L'))
im2, cdf = imtool.histeq(im)
figure()
hist(im2.flatten(), cdf.size)
show()
