# -*- coding: utf-8 -*-

"""
Copyright () 2018

All rights reserved

FILE: scipy_learn.py
AUTHOR: tianyuningmou
DATE CREATED:  @Time : 2018/3/19 下午2:46

DESCRIPTION:  .

VERSION: : #1 
CHANGED By: : tianyuningmou
CHANGE:  : 
MODIFIED: : @Time : 2018/3/19 下午2:46
"""

from PIL import Image
from numpy import *
from scipy.ndimage import filters, measurements, morphology


im = array(Image.open('fbb.jpeg').convert('L'))
im2 = filters.gaussian_filter(im, 5)
image_g = Image.fromarray(im2)
image_g.save('fbb_g.jpeg')

# Sobel导数滤波器
imx = zeros(im.shape)
filters.sobel(im, 1, imx)
image_x = Image.fromarray(uint8(imx))
image_x.save('fbb_x.jpeg')

imy = zeros(im.shape)
filters.sobel(im, 0, imy)
image_y = Image.fromarray(uint8(imy))
image_y.save('fbb_y.jpeg')

magnitude = sqrt(imx**2 + imy**2)
image_1 = Image.fromarray(uint8(magnitude))
image_1.save('fbb_1.jpeg')

#高斯导数滤波器
# 标准差
sigma = 1
imx = zeros(im.shape)
filters.gaussian_filter(im, (sigma, sigma), (0, 1), imx)
image_x = Image.fromarray(uint8(imx))
image_x.save('fbb_gx.jpeg')

imy = zeros(im.shape)
filters.gaussian_filter(im, (sigma, sigma), (1, 0), imy)
image_y = Image.fromarray(uint8(imy))
image_y.save('fbb_gy.jpeg')

# 载入图像，然后使用阈值化操作，以保证处理的图像为二值图像
im_2 = array(Image.open('fbb.jpeg').convert('L'))
im_2 = 1*(im_2<128)

labels, nbr_objects = measurements.label(im_2)
print('Numbers of objects:', nbr_objects)

# 形态学开操作，更好地分离各个对象
im_open = morphology.binary_opening(im_2, ones((9, 5)), iterations=2)
labels_open, nbr_objects_open = measurements.label(im_open)
print('Numbers of objects:', nbr_objects_open)
