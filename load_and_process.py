import pandas as pd
import cv2
import numpy as np
from skimage.feature import local_binary_pattern

dataset_path = 'fer2013/fer2013_clean2.csv'
image_size = (48, 48)
# settings for LBP
radius = 1  # LBP算法中范围半径的取值
n_points = 8 * radius  # 领域像素点数


# 载入数据
def load_fer2013():
    data = pd.read_csv(dataset_path)
    pixels = data['pixels'].tolist()  # -转为列表类型
    width, height = 48, 48
    faces = []  # -创建faces列表
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)  # -将pixels项转为48x48的二维矩阵
        face = cv2.resize(face.astype('uint8'), image_size)  # -将其中的数据转化为0-255之间的整数，并调整大小为64x64
        # face = local_binary_pattern(face, n_points, radius)
        faces.append(face.astype('float32'))  # -转化为float32类型
    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)  # -在数组的最后维度增加一个维度
    emotions = pd.get_dummies(data['emotion']).values  # -独热编码，将分类变量转换为虚拟/指标变量
    return faces, emotions


# 将数据归一化
def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x
