# -*- coding: utf-8 -*-
"""
需要安装的库：
    keras 2.2.4
    PyQt5 5.11.3
    pandas 0.24.2
    scikit-learn 0.21.2
    tensorflow 1.13.1
    imutils 0.5.2
    opencv-python 4.10.25

点击运行主程序runMain.py
"""


from EmotionRecongnition import Ui_MainWindow
from sys import argv,exit
from PyQt5.QtWidgets import QApplication, QMainWindow
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == '__main__':
    app = QApplication(argv)

    window = QMainWindow()
    ui = Ui_MainWindow(window)

    window.show()
    exit(app.exec_())