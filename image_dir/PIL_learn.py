# -*- coding: utf-8 -*-

"""
Copyright () 2018

All rights reserved

FILE: PIL_learn.py
AUTHOR: tianyuningmou
DATE CREATED:  @Time : 2018/3/15 下午2:14

DESCRIPTION:

VERSION: : #1 
CHANGED By: : tianyuningmou
CHANGE:  : 
MODIFIED: : @Time : 2018/3/15 下午2:14
"""

from PIL import Image

# 打开一张图片，并将其转为灰度图像
pil_image = Image.open('fbb.jpeg')
pil_image = pil_image.convert('L')
pil_image.save('fbb_L.jpeg')

# 创建缩略图
pil_image = Image.open('fbb.jpeg')
pil_image.thumbnail((128, 96))
pil_image.save('fbb_s.jpeg')

# 对图像某一块区域剪切并旋转，再放回去
pil_image = Image.open('fbb.jpeg')
box = (100, 0, 400, 300)
region = pil_image.crop(box)
region = region.transpose(Image.ROTATE_180)
pil_image.paste(region, box)
pil_image.save('fbb_c.jpeg')

# 调整尺寸和旋转
pil_image = Image.open('fbb.jpeg')
out = pil_image.resize((128, 128))
out = out.rotate(45)
out.save('fbb_r.jpeg')
