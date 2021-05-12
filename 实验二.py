# 神经网络相关包
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

# 画图相关包
import matplotlib.pyplot as plt

# 画图相关包
import numpy as np

# 从keras.datasets中导入数据集
# 训练数据集和测试数据集
# 训练数据集：图片x_train, 标签y_train
# 测试数据集：图片x_test, 标签y_test
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data('minist.npz')

#动态分配显存
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# 查看数据集格式
print(x_train.shape)          # 查看训练数据集图片的shape
print(y_train.shape)          # 查看训练数据集标签的shape
print(x_test.shape)           # 查看测试数据集图片的shape
print(y_test.shape)           # 查看测试数据集标签的shape

# 查看样本数据样式
print(x_train[1])        # 单个图像样本的数据形式
print(x_train[1].shape)  # 单个图像样本的数据shape
print(y_train[1])        # 单个标签样本的数据形式
print(y_train[1].shape)  # 单个标签样本的数据shape

# 查看标签
print(set(list(y_train)))
plt.imshow(x_train[0])
plt.show()

# 使用CNN网络时，需要将输入数据处理成shape为[hsize,wsize,channel]或者[channel,hsize,wsize]
# 该处的图像是黑白图像，只有一个channel

# 对训练图像数据reshape
x_train = x_train.reshape((-1,28,28,1))
print(x_train.shape)
# 对测试图像数据reshape
x_test = x_test.reshape((-1,28,28,1))
print(x_test.shape)