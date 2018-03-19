# -*- coding: utf-8 -*-

"""
Copyright () 2018

All rights reserved

FILE: rof_model.py
AUTHOR: tianyuningmou
DATE CREATED:  @Time : 2018/3/19 下午4:11

DESCRIPTION:  .

VERSION: : #1 
CHANGED By: : tianyuningmou
CHANGE:  : 
MODIFIED: : @Time : 2018/3/19 下午4:11
"""

from numpy import *
from numpy import random
from scipy.ndimage import filters
from scipy.misc import imsave


def denoise(im, U_init, tolerance=0.1, tau=0.125, tv_weight=100):
    """
        使用 A.Chambolle（2005）在公式（11）中的计算步骤实现 Rudin-Osher-Fatemi（ROF）去噪模型
        输入：含有噪声的输入图像（灰度图像）、 U 的初始值、 TV 正则项权值、步长、停业条件
        输出：去噪和去除纹理后的图像、纹理残留
    """
    # 噪声图像的大小
    m, n = im.shape
    # 初始化
    U = U_init
    # 对偶域的x分量
    Px = im
    # 对偶域的y分量
    Py = im
    error = 1

    while(error > tolerance):
        Uold = U

        # 原始变量的梯度
        GrandUx = roll(U, -1, axis=1)-U
        GrandUy = roll(U, -1, axis=0)-U

        # 更新对偶变量
        PxNew = Px + (tau/tv_weight)*GrandUx
        PyNew = Py + (tau/tv_weight)*GrandUy
        NormNew = maximum(1, sqrt(PxNew**2 + PyNew**2))

        # 更新对偶分量
        Px = PxNew/NormNew
        Py = PyNew/NormNew

        # 更新原始变量
        RxPx = roll(Px, 1, axis=1)
        RyPy = roll(Py, 1, axis=0)

        # 对偶域的散du
        DivP = (Px - RxPx) + (Py - RyPy)
        # 更新原始变量
        U = im + tv_weight*DivP

        # 更新误差
        error = linalg.norm(U-Uold)/sqrt(n*m)

    # 去噪后的图像和纹理残余
    return U, im-U


# 使用噪声创建合成图像
im = zeros((500, 500))
im[100: 400, 100: 400] = 128
im[200: 300, 200: 300] = 255
im = im + 30*random.standard_normal((500, 500))

U, T = denoise(im, im)
G = filters.gaussian_filter(im, 10)

# 保存生成结果
imsave('synth_rof.pdf', U)
imsave('synth_gaussian.pdf', G)
