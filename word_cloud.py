# -*- coding: utf-8 -*-

"""
Copyright () 2018

All rights reserved

FILE: word_cloud.py
AUTHOR: tianyuningmou
DATE CREATED:  @Time : 2018/3/28 上午10:58

DESCRIPTION:  .

VERSION: : #1 
CHANGED By: : tianyuningmou
CHANGE:  : 
MODIFIED: : @Time : 2018/3/28 上午10:58
"""

from matplotlib import pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator
from scipy.misc import imread
import jieba
import codecs

# 读取文本
text = codecs.open('fbb.txt', 'r').read()
# 解析背景图片
bg_pic = imread('fbb.jpg')
# 生成词云
word_cloud = WordCloud(mask=bg_pic, background_color='white', max_words=500, scale=1.5, font_path='xg.ttf').generate(text)
# 背景图
image_color = ImageColorGenerator(bg_pic)
# 显示词云
plt.imshow(word_cloud)
plt.axis('off')
plt.show()
