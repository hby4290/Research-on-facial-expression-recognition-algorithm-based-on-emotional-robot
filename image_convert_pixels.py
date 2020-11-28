from PIL import Image
import cv2
import pandas as pd
import numpy as np
import os
import csv
from skimage.feature import local_binary_pattern

file = 'fer2013/fer2013_clean2_lbp.csv'
data = pd.read_csv(file)
image_data = data.loc[0][1]
image_array = list(image_data.split())
image_array = np.asarray(image_array)
image = image_array.reshape(48, 48).astype('float64')
Image.fromarray(image.astype('uint8'), mode='L').save('0.png')
# img = Image.open('../fer2013/pic_clean_png/Training/anger/0.png')
# img2 = cv2.imread('../fer2013/pic_clean_png/Training/anger/1.png', 0)
# # img = img.convert('L')
# # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# pixels = []
# pixels2 = []
# for i in range(img.size[0]):
#     for j in range(img.size[1]):
#         pixels.append(img.getpixel((j, i)))
# for i in range(img2.shape[0]):
#     for j in range(img2.shape[1]):
#         pixels2.append(img2.item(i, j))
# print(pixels)
# print(pixels2)
# print(img2.dtype, img2.shape[0], img2.shape[1], img2.shape, img.size)

# data = pd.read_csv('fer2013/fer2013_clean.csv')
# # data.insert(3, 'num', '')
# imgList = os.listdir('../fer2013/pic_clean/PrivateTest/surprise')
# # imgList.sort()
# for imgName in imgList:
#     imgName = imgName.replace('.jpg', '')
#     if imgName.find('x'):
#         imgName = imgName.replace('x', '')
#     data.loc[[int(imgName)], ['num']] = imgName
# data.to_csv('fer2013/fer2013_clean.csv', index=False)

# # settings for LBP
# radius = 1  # LBP算法中范围半径的取值
# n_points = 8 * radius  # 领域像素点数
#
# file = open('fer2013/fer2013_clean2_lbp.csv', 'a', newline='')
# writer = csv.writer(file)
# directory = '../fer2013/pic_clean_png/Training/anger'
# imgList = os.listdir(directory)
# for imgName in imgList:
#     imgPath = os.path.join(directory, imgName)
#     image = cv2.imread(imgPath, 0)
#     image_lbp = local_binary_pattern(image, n_points, radius)
#     pixel = ''
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             pixel = pixel + str(image_lbp.item(i, j)) + ' '
#     pixel = pixel.rstrip()
#     writer.writerow(['0', pixel, 'Training'])
# file.close()
