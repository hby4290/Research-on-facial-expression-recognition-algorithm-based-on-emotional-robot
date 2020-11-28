import cv2
import pandas as pd
import numpy as np
from PIL import Image
import os
import csv

emotions = {
    '0': 'anger',  # 生气
    '1': 'disgust',  # 厌恶
    '2': 'fear',  # 恐惧
    '3': 'happy',  # 开心
    '4': 'sad',  # 伤心
    '5': 'surprise',  # 惊讶
    '6': 'neutral',  # 中性
}

file = 'fer2013/fer2013.csv'
directory = '../fer2013/pic_clean_alt'
detection_model_path = 'haarcascade_files/haarcascade_frontalface_alt.xml'
face_detection = cv2.CascadeClassifier(detection_model_path)


# 创建文件夹
def createDir(dir):
    if os.path.exists(dir) is False:
        os.makedirs(dir)


imageCount = 0
dropCount = 0
new_file = open('fer2013/fer2013_alt.csv', 'w', newline='')
write = csv.writer(new_file)
write.writerow(['emotion', 'pixels', 'Usage'])
data = pd.read_csv(file)
createDir(directory)
for index in range(len(data)):
    # 解析每一行csv文件内容
    emotion_data = data.loc[index][0]
    image_data = data.loc[index][1]
    usage_data = data.loc[index][2]
    # 将图片数据转换成48*48
    image_array = list(image_data.split())
    image_array = np.asarray(image_array)
    image = image_array.reshape(48, 48)
    # 选择分类，并创建文件名
    dirName = os.path.join(directory, usage_data)
    emotionName = emotions[str(emotion_data)]
    # 图片要保存的文件夹
    imagePath = os.path.join(dirName, emotionName)
    # 创建“用途文件夹”和“表情”文件夹
    createDir(dirName)
    createDir(imagePath)
    # 图片文件名
    imageName = os.path.join(imagePath, str(index) + '.jpg')

    Image.fromarray(image.astype('uint8'), mode='L').save(imageName)
    img = cv2.imread(imageName)
    faces = face_detection.detectMultiScale(img, scaleFactor=1.1,
                                            minNeighbors=5, minSize=(30, 30),
                                            flags=cv2.CASCADE_SCALE_IMAGE)
    if len(faces) > 0:
        write.writerow([emotion_data, image_data, usage_data])
        drop_image = os.path.join(imagePath, str(index) + 'x.jpg')
        os.rename(imageName, drop_image)
        dropCount = dropCount + 1
    imageCount = index
# imgCount = imageCount - dropCount
print('总共有' + str(imageCount) + '张图片')
print('检测了' + str(dropCount) + '张图片')
# print('剩余' + str(imgCount) + '张图片')

