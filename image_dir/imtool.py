# -*- coding: utf-8 -*-

"""
Copyright () 2018

All rights reserved

FILE: imtool.py
AUTHOR: tianyuningmou
DATE CREATED:  @Time : 2018/3/16 下午4:41

DESCRIPTION:  .

VERSION: : #1 
CHANGED By: : tianyuningmou
CHANGE:  : 
MODIFIED: : @Time : 2018/3/16 下午4:41
"""

from PIL import Image
import numpy as np

# 图像缩放函数
def imresize(im, sz):
    pil_im = Image.fromarray(np.uint8(im))
    return np.array(pil_im.resize(sz))

# 对一幅灰度图像进行直方图均衡化
def histeq(im, nbr_bins=256):
    # 计算图像的直方图
    im_hist, bins = np.histogram(im.flatten(), nbr_bins, normed=True)
    # cumulative distribution functions
    cdf = im_hist.cumsum()
    # 归一化
    cdf = 255 * cdf / cdf[-1]
    # 利用累积分布函数的线性插值，计算新的像素值
    im2 = np.interp(im.flatten(), bins[:-1], cdf)
    return im2.reshape(im.shape), cdf
