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


# 计算图像列表的平均图像
def compute_average(im_list):
    # 打开第一幅图像，将其存储在浮点型数组中
    average_im = np.array(Image.open(im_list[0]), 'f')
    fail_num = 0
    for im_name in im_list[1:]:
        try:
            average_im += np.array(Image.open(im_name))
        except:
            print(im_name + '...skipped')
            fail_num += 1
    average_im /= (len(im_list) - fail_num)
    # 返回uint8类型的平均图像
    return np.array(average_im, 'uint8')


# PCA
def pca(X):
    """
    主成分分析
    :param X: 矩阵x，其中该矩阵中存储训练数据每一行为一条训练数据
    :return: 投影矩阵、方差和均值
    """
    # 获取维数
    num_data, dim = X.shape
    # 数据中心化
    mean_X = X.mean(axis=0)
    X = X - mean_X
    # PCA - 使用紧凑技巧
    if dim > num_data:
        # 协方差矩阵
        M = np.dot(X, X.T)
        # 特征值和特征向量
        e, EV = np.linalg.eigh(M)
        # 紧凑技巧
        tmp = np.dot(X.T, EV).T
        # 由于最后的特征向量是我们所需要的，所以需要将其逆转
        V = tmp[::-1]
        # 由于特征值是按照递增顺序排列的，所以需要将其逆转
        S = np.sqrt(e)[::-1]
        for i in range(V.shape[1]):
            V[:, i] /= S
    # PCA - 使用SVD方法
    else:
        U, S, V = np.linalg.svd(X)
        V = V[:num_data]
    return V, S, mean_X
