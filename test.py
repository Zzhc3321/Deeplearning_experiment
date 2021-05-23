# 神经网络
import tensorflow.keras.layers as layers
import tensorflow.keras as keras
import tensorflow as tf
# 画图包
import matplotlib.pyplot as plt

# 数值计算
import numpy as np

# tensorflow_datasets 包
import tensorflow_datasets as tfds

# print(tf.__version__)

# #动态分配显存
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# 使用load方法载入数据
dataset, info = tfds.load('imdb_reviews/subwords8k',data_dir='data', download=False,with_info=True,as_supervised=True)

# 训练集测试集
train_dataset, test_dataset = dataset['train'], dataset['test']

print(train_dataset)