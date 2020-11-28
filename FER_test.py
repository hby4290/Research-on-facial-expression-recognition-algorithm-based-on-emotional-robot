# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\大学\毕业设计\Emotion_Recognition\FER_test.ui'
#
# Created by: PyQt5 UI code generator 5.13.2
#
# WARNING! All changes made in this file will be lost!

import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QMovie
import cv2
import imutils
import numpy as np
import time
from os import remove
from slice_png import img as bgImg
from base64 import b64decode
from keras.models import load_model
from keras.preprocessing.image import img_to_array


class Ui_FER_test(object):
    def __init__(self, Window):
        self.setupUi(Window)
        self.retranslateUi(Window)
        self.timer_camera = QtCore.QTimer()  # 定时器
        self.cap = cv2.VideoCapture()  # 屏幕画面对象
        self.CAM_NUM = 0  # 摄像头标号
        self.model_path = 'models/48.20_my_XCEPTION.42-0.69.hdf5'  # 模型路径
        self.slot_init()  # 槽函数设置

    def setupUi(self, FER_test):
        FER_test.setObjectName("FER_test")
        FER_test.setWindowModality(QtCore.Qt.NonModal)
        FER_test.resize(820, 346)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("D:\\大学\\毕业设计\\Emotion_Recognition\\images_test/result.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        FER_test.setWindowIcon(icon)
        self.label_outputResult = QtWidgets.QLabel(FER_test)
        self.label_outputResult.setGeometry(QtCore.QRect(480, 70, 300, 250))
        self.label_outputResult.setMinimumSize(QtCore.QSize(300, 250))
        self.label_outputResult.setMaximumSize(QtCore.QSize(300, 250))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(18)
        self.label_outputResult.setFont(font)
        self.label_outputResult.setObjectName("label_outputResult")
        self.label_face = QtWidgets.QLabel(FER_test)
        self.label_face.setGeometry(QtCore.QRect(20, 30, 420, 300))
        self.label_face.setMinimumSize(QtCore.QSize(420, 300))
        self.label_face.setMaximumSize(QtCore.QSize(420, 300))
        self.label_face.setText("")
        self.label_face.setObjectName("label_face")
        self.label_useTime = QtWidgets.QLabel(FER_test)
        self.label_useTime.setGeometry(QtCore.QRect(480, 20, 61, 41))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(12)
        self.label_useTime.setFont(font)
        self.label_useTime.setObjectName("label_useTime")
        self.label_time = QtWidgets.QLabel(FER_test)
        self.label_time.setGeometry(QtCore.QRect(542, 20, 71, 41))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(12)
        self.label_time.setFont(font)
        self.label_time.setObjectName("label_time")
        self.label_scanResult = QtWidgets.QLabel(FER_test)
        self.label_scanResult.setGeometry(QtCore.QRect(620, 20, 91, 41))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(12)
        self.label_scanResult.setFont(font)
        self.label_scanResult.setObjectName("label_scanResult")
        self.label_result = QtWidgets.QLabel(FER_test)
        self.label_result.setGeometry(QtCore.QRect(720, 20, 81, 41))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(12)
        self.label_result.setFont(font)
        self.label_result.setObjectName("label_result")

        self.retranslateUi(FER_test)
        QtCore.QMetaObject.connectSlotsByName(FER_test)

    def retranslateUi(self, FER_test):
        _translate = QtCore.QCoreApplication.translate
        FER_test.setWindowTitle(_translate("FER_test", "FER_test"))
        self.label_outputResult.setText(_translate("FER_test", "表情识别概率显示区"))
        self.label_useTime.setText(_translate("FER_test", "用时："))
        self.label_time.setText(_translate("FER_test", "0s"))
        self.label_scanResult.setText(_translate("FER_test", "识别结果："))
        self.label_result.setText(_translate("FER_test", "None"))

    def slot_init(self):  # 定义槽函数
        self.open_camera()
        self.timer_camera.timeout.connect(self.show_camera)

    def open_camera(self):
        if not self.timer_camera.isActive():  # 检查定时状态
            flag = self.cap.open(self.CAM_NUM)  # 检查相机状态
            if not flag:  # 相机打开失败提示
                msg = QtWidgets.QMessageBox.warning(self.FER_test, u"Warning",
                                                    u"请检测相机与电脑是否连接正确！ ",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                # 准备运行识别程序
                QtWidgets.QApplication.processEvents()
                self.label_face.setText('正在启动识别系统...\n\nleading')
                # 新建对象
                self.emotion_model = Emotion_Rec(self.model_path)
                QtWidgets.QApplication.processEvents()
                # 打开定时器
                self.timer_camera.start(30)
        else:
            # 定时器未开启，界面回复初始状态
            self.timer_camera.stop()
            self.cap.release()
            self.label_face.clear()
            self.label_outputResult.clear()
            self.label_result.setText('None')
            self.label_time.setText('0 s')

    def show_camera(self):
        # 定时器槽函数，每隔一段时间执行
        flag, self.image = self.cap.read()  # 获取画面
        self.image = cv2.flip(self.image, 1)  # 左右翻转

        tmp = open('slice.png', 'wb')
        tmp.write(b64decode(bgImg))
        tmp.close()
        canvas = cv2.imread('slice.png')  # 用于数据显示的背景图片
        remove('slice.png')

        time_start = time.time()  # 计时
        # 使用模型预测
        result = self.emotion_model.run(self.image, canvas, self.label_face, self.label_outputResult)
        time_end = time.time()
        # 在界面显示结果
        self.label_result.setText(result)
        self.label_time.setText(str(round((time_end-time_start),3))+'s')


class Emotion_Rec:
    def __init__(self, model_path):

        # 载入数据和图片的参数
        detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'

        emotion_model_path = model_path

        # 载入人脸检测模型
        self.face_detection = cv2.CascadeClassifier(detection_model_path)  # 级联分类器

        # 载入人脸表情识别模型
        self.emotion_classifier = load_model(emotion_model_path, compile=False)
        # 表情类别
        self.EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    def run(self, frame_in, canvas, label_face, label_result):
        # frame_in 摄像画面或图像
        # canvas 用于显示的背景图
        # label_face 用于人脸显示画面的label对象
        # label_result 用于显示结果的label对象

        # 调节画面大小
        frame = imutils.resize(frame_in, width=300)  # 缩放画面
        # frame = cv2.resize(frame, (300,300))  # 缩放画面
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转为灰度图

        # 检测人脸
        faces = self.face_detection.detectMultiScale(gray, scaleFactor=1.1,
                                                     minNeighbors=5, minSize=(30, 30),
                                                     flags=cv2.CASCADE_SCALE_IMAGE)
        preds = []  # 预测的结果
        label = None  # 预测的标签
        (fX, fY, fW, fH) = None, None, None, None  # 人脸位置
        if len(faces) > 0:
            # 选择检测到的ROI最大的人脸
            faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces

            # 从灰度图中提取感兴趣区域（ROI），将其大小转换为64*64 pixels，并为通过CNN的分类器准备ROI
            roi = gray[fY-5:fY + fH+5, fX-5:fX + fW+5]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # 用模型预测各分类的概率
            preds = self.emotion_classifier.predict(roi)[0]
            # emotion_probability = np.max(preds)  # 最大的概率
            label = self.EMOTIONS[preds.argmax()]  # 选取最大概率的表情类

        frameClone = frame.copy()  # 复制画面
        # canvas = 255* np.ones((250, 300, 3), dtype="uint8")
        # canvas = cv2.imread('slice.png', flags=cv2.IMREAD_UNCHANGED)

        for (i, (emotion, prob)) in enumerate(zip(self.EMOTIONS, preds)):
            # 用于显示各类别概率
            text = "{}: {:.2f}%".format(emotion, prob * 100)

            # 绘制表情类和对应概率的条形图
            w = int(prob * 300)+7
            cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 192, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)

            # 圈出人脸区域并显示识别结果
            cv2.putText(frameClone, label, (fX, fY - 10),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 255, 0), 1)
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (255, 255, 0), 1)

        # 调整画面大小与界面相适应
        # if (frameClone.shape[0]/frameClone.shape[1]) > (280/420):
        #     frameClone = imutils.resize(frameClone, height=280)
        # else:
        #     frameClone = imutils.resize(frameClone, width=420)
        frameClone = cv2.resize(frameClone, (420, 280))

        # 在Qt界面中显示人脸
        show = cv2.cvtColor(frameClone, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        label_face.setPixmap(QtGui.QPixmap.fromImage(showImage))
        QtWidgets.QApplication.processEvents()

        # 在显示结果的label中显示结果
        show = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        label_result.setPixmap(QtGui.QPixmap.fromImage(showImage))

        return(label)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    window = QWidget()
    ui = Ui_FER_test(window)

    window.show()
    exit(app.exec_())