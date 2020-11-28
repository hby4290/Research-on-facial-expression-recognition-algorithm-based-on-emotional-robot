"""
Description: 训练人脸表情识别程序
"""

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from load_and_process import load_fer2013
from load_and_process import preprocess_input
from models.cnn import mini_XCEPTION
from models.cnn import tiny_XCEPTION
from models.cnn import my_XCEPTION
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import matplotlib.pyplot as plt

# 参数
batch_size = 32
num_epochs = 10000
input_shape = (48, 48, 1)
validation_split = .2
verbose = 1
num_classes = 7
patience = 40
base_path = 'models/'


# 构建模型
# model = my_XCEPTION(input_shape, num_classes)
model = mini_XCEPTION(input_shape, num_classes)
# model = tiny_XCEPTION(input_shape, num_classes)
model.compile(optimizer='adam',  # 优化器采用adam
              loss='categorical_crossentropy',  # 交叉熵损失函数
              metrics=['accuracy'])
model.summary()


# 定义回调函数 Callbacks 用于训练过程
log_file_path = base_path + '48.28mini_emotion_training.log'
csv_logger = CSVLogger(log_file_path, append=True)
early_stop = EarlyStopping('val_loss', patience=patience)    # patience次不变则停止训练
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                              patience=int(patience/4),
                              verbose=1)
# 模型位置及命名
trained_models_path = base_path + '48.28_mini_XCEPTION'
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'

# 定义模型权重位置、命名等
model_checkpoint = ModelCheckpoint(model_names,
                                   'val_loss', verbose=1,
                                   save_best_only=True)
callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]


# 载入数据集
faces, emotions = load_fer2013()
faces = preprocess_input(faces)
num_samples, num_classes = emotions.shape   # -返回emotions维度大小的元组

# 划分训练、测试集
xtrain, xtest, ytrain, ytest = train_test_split(faces, emotions, test_size=0.2, shuffle=True)   # -20%的数据用于测试

# 图片产生器，在批量中对数据进行增强，扩充数据集大小
data_generator = ImageDataGenerator(
                        featurewise_center=False,  # 布尔。将输入平均值设置为 0，按特征进行。
                        featurewise_std_normalization=False,  # 布尔。按数据集的当前方向划分输入。
                        rotation_range=10,  # 随机旋转的度范围。
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)  # 布尔。随机水平翻转输入。

# 利用数据增强进行训练
model.fit_generator(data_generator.flow(xtrain, ytrain, batch_size),
                    steps_per_epoch=len(xtrain) / batch_size,
                    epochs=num_epochs,
                    verbose=1, callbacks=callbacks,
                    validation_data=(xtest, ytest))


# 显示预测的混淆矩阵
# 计算预测
predictions = model.predict(xtest, batch_size=32)
y_pred = [np.argmax(probas) for probas in predictions]
y_true = ytest.argmax(axis=-1)
class_names = ('anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')


def plot_confusion_matrix(cm, classes, title="Confusion matrix", cmap=plt.cm.Blues):
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    # plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = ".2f"
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()


# 计算混淆矩阵
cnf_matrix = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title="Normalized confusion matrix")
plt.show()
