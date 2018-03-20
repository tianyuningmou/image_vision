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


def get_descriptors(image, filtered_coords, wid=5):
    # 对于每个返回的点，返回点周围2*wid+1个像素的值（假设选取点min_distance>wid）
    desc = []
    for coords in filtered_coords:
        patch = image[coords[0]-wid: coords[0]+wid+1, coords[1]-wid: coords[1]+wid+1].flatten()
        desc.append(patch)
    return desc


def match(desc1, desc2, threshold=0.5):
    # 对于第一个图像中的每个角点描述子，使用归一化互相关，选取它在第二幅图像中的匹配角点
    n = len(desc1[0])
    # 点对的距离
    d = -ones((len(desc1), len(desc2)))
    for i in range(len(desc1)):
        for j in range(len(desc2)):
            d1 = (desc1[i] - mean(desc1[i])) / std(desc1[i])
            d2 = (desc2[j] - mean(desc2[j])) / std(desc2[j])
            ncc_value = sum(d1 * d2) / (n - 1)
            if ncc_value > threshold:
                d[i, j] = ncc_value
    ndx = argsort(-d)
    matchscores = ndx[:, 0]
    return matchscores


def match_twosided(desc1, desc2, threshold=0.5):
    # 两边对称版本的match
    matches_12 = match(desc1, desc2, threshold)
    matches_21 = match(desc2, desc1, threshold)
    ndx_12 = where(matches_12 >= 0)[0]
    # 去除非对称的匹配
    for n in ndx_12:
        if matches_21[matches_12[n]] != n:
            matches_12[n] = -1
    return matches_12


def appendimage(im1, im2):
    # 将两张图像拼接成一张
    row1 = im1.shape[0]
    row2 = im2.shape[0]
    if row1 < row2:
        im1 = concatenate((im1, zeros((row2 - row1, im1.shape[1]))), axis=0)
    elif row1 > row2:
        im2 = concatenate((im2, zeros((row1 - row2, im2.shape[1]))), axis=0)
    return concatenate((im1, im2), axis=1)


def plot_matches(im1, im2, locs1, locs2, matchscores, show_below=True):
    """
        显示一幅带有连接匹配之间连线的图片
        输入：im1, im2(数组图像), locs1, locs2(特征位置), matchscores(match()的输出), show_below(如果图像应该显示在匹配的下方)
    """
    im3 = appendimage(im1, im2)
    if show_below:
        im3 = vstack((im3, im3))
    imshow(im3)
    cols1 = im1.shape[1]
    for i, m in enumerate(matchscores):
        if m > 0:
            plot([locs1[i][1], locs2[m][1] + cols1], [locs1[i][0], locs2[m][0]], 'c')
    axis('off')


def plot_harris_points(image, filtered_coords):
    # 绘制图像中检测到的角点
    figure()
    gray()
    imshow(image)
    plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords], '*')
    axis('off')
    show()


if __name__ == '__main__':
    wid = 5
    im1 = array(Image.open('fbb.jpeg').convert('L'))
    harrisim1 = compute_harris_response(im1, 5)
    filtered_coords1 = get_harris_points(harrisim1, wid+1)
    d1 = get_descriptors(im1, filtered_coords1, wid)

    im2 = array(Image.open('zyq.jpeg').convert('L'))
    harrisim2 = compute_harris_response(im2, 5)
    filtered_coords2 = get_harris_points(harrisim2, wid+1)
    d2 = get_descriptors(im2, filtered_coords2, wid)

    print('start match')
    matches = match_twosided(d1, d2)
    figure()
    gray()
    plot_matches(im1, im2, filtered_coords1, filtered_coords2, matches)
    show()
