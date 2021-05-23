# 神经网络相关包
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorboardX import SummaryWriter


# writer = SummaryWriter(comment='test')

# 从keras.datasets中导入数据集

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

# # 查看标签
# print(set(list(y_train)))
# plt.imshow(x_train[0])
# plt.show()

# 使用CNN网络时，需要将输入数据处理成shape为[hsize,wsize,channel]或者[channel,hsize,wsize]
# 该处的图像是黑白图像，只有一个channel

# 对训练图像数据reshape
x_train = x_train.reshape((-1,28,28,1))
print(x_train.shape)
# 对测试图像数据reshape
x_test = x_test.reshape((-1,28,28,1))
print(x_test.shape)



# 构造网络
model = keras.Sequential()

# 卷积层Conv2D
# 参数：
# input_shape：输入图像数据的[rows,cols,channels]，不包括batch_size。
# filters:卷积核个数
# kernel_size:卷积核大小
# strides：步长
# padding：填充方式，此处是valid填充。valid:表示不够卷积核大小的块,则丢弃;same表示不够卷积核大小的块就补0,所以输出和输入形状相同
# activation：激活函数，此处是relu函数
model.add(layers.Conv2D(input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]),
                        filters=32, kernel_size=(3,3), strides=(1,1), padding='valid',
                        activation='relu'))

# 池化层：MaxPool2D
#	参数：
# pool_size：池化大小
model.add(layers.MaxPool2D(pool_size=(2,2)))

# Flatten层，将上一层的输出拉平成一维向量
model.add(layers.Flatten())

# 全连接层
# 隐藏层：包含32个隐藏神经元，激活函数是relu
model.add(layers.Dense(32, activation='relu'))
# 输出层：包含10个输出神经元，激活函数是softmax，每个神经元的输出值表示输入属于该类的概率
model.add(layers.Dense(10, activation='softmax'))


#模型训练方法配置
model.compile(optimizer=keras.optimizers.Adam(),               # 使用Adam作为优化器
             loss=keras.losses.SparseCategoricalCrossentropy(),# 多分类问题，使用交叉熵损失函数
             metrics=['accuracy'])                             # 使用准确率作为评估指标


# 查看模型细节
model.summary()




# 模型训练
# 输入参数：
# x_train：训练样本的输入
# y_train：训练样本的目标输出
# epochs：全量样本的训练轮数
# batch_size：默认值为32
# validation_split：验证数据集占训练样本的比例

# 输出：
# 模型每一次训练时在验证集、训练集上的metrics配置的指标,loss指标默认存在metrics中
history = model.fit(x_train, y_train, batch_size=64, epochs=50, validation_split=0.1)


# # 模型训练结果打印
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.legend(['training', 'valivation'], loc='upper left')
# plt.show()
#
#
# # 模型训练结果打印
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.legend(['training_loss', 'valivation_loss'], loc='upper left')
# plt.show()