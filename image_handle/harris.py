# -*- coding: utf-8 -*-

"""
Copyright () 2018

All rights reserved

FILE: harris.py
AUTHOR: tianyuningmou
DATE CREATED:  @Time : 2018/3/19 下午4:58

DESCRIPTION:  .

VERSION: : #1 
CHANGED By: : tianyuningmou
CHANGE:  : 
MODIFIED: : @Time : 2018/3/19 下午4:58
"""

"""
    Harris角点检测算法：如果像素周围显示存在多于一个方向的边，我们认为改点为兴趣点，即角点。
"""

from PIL import Image
from pylab import *
from scipy.ndimage import filters


def compute_harris_response(im, sigma=3):
    # 在一幅图像中，对每个像素计算Harris角点检测器响应次数
    # 计算导数
    imx = zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (0, 1), imx)
    imy = zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (1, 0), imy)

    # 计算Harris矩阵的分量
    Wxx = filters.gaussian_filter(imx * imx, sigma)
    Wxy = filters.gaussian_filter(imx * imy, sigma)
    Wyy = filters.gaussian_filter(imy * imy, sigma)

    # 计算特征值和迹
    Wdet = Wxx * Wyy - Wxy ** 2
    Wtr = Wxx + Wyy

    return Wdet/Wtr


def get_harris_points(harrisim, min_dist=10, threshold=0.1):
    # 从一幅Harris响应图像中返回角点，min_dist为分割角点和图像边界的最小像素数目
    # 寻找高于阈值的候选角点
    corner_threshold = harrisim.max() * threshold
    harrisim_t = (harrisim > corner_threshold) * 1

    # 得到候选点的坐标
    coords = array(harrisim_t.nonzero()).T

    # 得到候选点的Harris响应值
    candidate_values = [harrisim[c[0], c[1]] for c in coords]

    # 对候选点按照Harris响应值进行排序
    index = argsort(candidate_values)

    # 将可行点的位置保存在数组中
    allowed_location = zeros(harrisim.shape)
    allowed_location[min_dist: -min_dist, min_dist: -min_dist] = 1

    # 按照min_distance原则，选择最佳Harris点
    filtered_coords = []
    for i in index:
        if allowed_location[coords[i, 0], coords[i, 1]] == 1:
            filtered_coords.append(coords[i])
            allowed_location[(coords[i, 0] - min_dist): (coords[i, 0] + min_dist),
                             (coords[i, 1] - min_dist): (coords[i, 1] + min_dist)] = 0
    return filtered_coords


def plot_harris_points(image, filtered_coords):
    # 绘制图像中检测到的角点
    figure()
    gray()
    imshow(image)
    plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords], '*')
    axis('off')
    show()


if __name__ == '__main__':
    im = array(Image.open('fbb.jpeg').convert('L'))
    harrisim = compute_harris_response(im)
    filtered_coords = get_harris_points(harrisim, 6)
    plot_harris_points(im, filtered_coords)
