# -*- coding: utf-8 -*-

"""
Copyright () 2018

All rights reserved

FILE: matplotlib_learn.py
AUTHOR: tianyuningmou
DATE CREATED:  @Time : 2018/3/15 下午2:39

DESCRIPTION:  .

VERSION: : #1 
CHANGED By: : tianyuningmou
CHANGE:  : 
MODIFIED: : @Time : 2018/3/15 下午2:39
"""

from PIL import Image
from matplotlib.pylab import *

p1_image = array(Image.open('fbb.jpeg'))
imshow(p1_image)

x = [100, 100, 400, 400]
y = [200, 500, 200, 500]

plot(x, y, 'r*')
plot(x[:2], y[:2])
title('Plotting: "fbb"')
# 使坐标轴不显示
axis('off')
show()

p2_image = array(Image.open('fbb.jpeg').convert('L'))
figure()
gray()
contour(p2_image, origin='image')
axis('equal')
axis('off')
show()

p3_image = array(Image.open('fbb.jpeg').convert('L'))
figure()
hist(p3_image.flatten(), 128)
show()

p4_image = array(Image.open('fbb.jpeg'))
imshow(p4_image)
print('Please click 3 points')
x = ginput(3)
print('You clicked: ', x)
show()
