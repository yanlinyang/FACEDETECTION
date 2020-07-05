from PyQt5.QtCore import *

from PyQt5.QtWidgets import *

from PyQt5.QtGui import *
import numpy as np
import cv2,os,time,csv
from skimage import io
import dlib         # 人脸处理的库 Dlib

from PIL import Image, ImageDraw, ImageFont
import pandas as pd
# Dlib 正向人脸检测器




import csv

with open('yly/data/person_all.csv', 'r') as f:
    reader = csv.reader(f)
    result = np.array(list(reader))
    print(result[:,0])

from PIL import Image,ImageDraw,ImageFont
a = '我们不一样' # 定义文本
font = ImageFont.truetype("simsun.ttc", 30) # 定义字体，这是本地自己下载的
img = Image.new('RGB',(300,300),(255,180,0)) # 新建长宽300像素，背景色为（255,180,0）的画布对象
draw = ImageDraw.Draw(img) # 新建画布绘画对象
draw.text( (50,50), a,(255,0,0),font=font) # 在新建的对象 上坐标（50,50）处开始画出红色文本
# 左上角为画布坐标（0,0）点
img.show()
img.save('img.jpeg')
