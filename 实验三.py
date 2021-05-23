import tensorflow.keras.layers as layers
import tensorflow.keras as keras
import tensorflow as tf
import tensorflow_datasets as tfds


# 注意，此时使用的padding方式是按照batch padding，因此不能使用函数式API的方式构建模型
# 因为此方式中要指定输入的shape，但是输入的shape不固定
# 因此使用Sequential的方式构建模型
def LSTM_model():
    model = keras.Sequential([
        # --------------embedding层--------------------- #
        # embedding layer 参数
        # input_dim：词汇表的大小
        # output_dim：词向量的维度
        # mask_zero：是否把0当成是应该被遮蔽的特殊padding值。默认为false
        # input_length：输入序列的长度，如果是固定的话则需要。
        layers.Embedding(input_dim=tokenizer.vocab_size, output_dim=embedding_size),
        # ---------------双层双向LSTM层----------------- #
        # Bidirectional layer参数
        # layer: Recurrent 实例。此处为LSTM。
        # merge_mode: 前向和后向 RNN 的输出的结合模式。 为 {'sum', 'mul', 'concat'（默认）, 'ave', None} 其中之一。 如果是 None，输出不会被结合，而是作为一个列表被返回。

        # LSTM layer 参数
        # unit：正整数，输出空间的维度。
        # input_dim：输入的维度（整数）。 将此层用作模型中的第一层时，此参数（或者，关键字参数 input_shape）是必需的。
        # input_length: 输入序列的长度，在恒定时指定。
        # return_sequences:布尔值。是返回输出序列中的最后一个输出，还是全部序列。
        # return_state: 布尔值。除了输出之外是否返回最后一个状态。
        layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
        # layers.Bidirectional(layers.LSTM(32)),

        # --------------全连接层--------------- #
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


# 两层双向GRU模型
def GRU_model():
    model = tf.keras.Sequential([
        # --------------embedding层--------------------- #
        # embedding layer 参数
        # input_dim：词汇表的大小
        # output_dim：词向量的维度
        # mask_zero：是否把0当成是应该被遮蔽的特殊padding值。默认为false
        # input_length：输入序列的长度，如果是固定的话则需要。
        layers.Embedding(input_dim=tokenizer.vocab_size, output_dim=embedding_size),

        # ---------------双层双向GRU层----------------- #
        # Bidirectional layer参数
        # layer: Recurrent 实例。此处为GRU。
        # merge_mode: 前向和后向 GRU 的输出的结合模式。 为 {'sum', 'mul', 'concat'（默认）, 'ave', None} 其中之一。 如果是 None，输出不会被结合，而是作为一个列表被返回。

        # GRU layer 参数
        # unit：正整数，输出空间的维度。
        # input_dim：输入的维度（整数）。 将此层用作模型中的第一层时，此参数（或者，关键字参数 input_shape）是必需的。
        # input_length: 输入序列的长度，在恒定时指定。
        # return_sequences:布尔值。是返回输出序列中的最后一个输出，还是全部序列。
        layers.Bidirectional(layers.GRU(32, return_sequences=True)),
        # layers.Bidirectional(layers.GRU(32)),

        # --------------全连接层--------------- #
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


if __name__ == "__main__":

    #动态分配显存
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


    log_dir_lstm="logs/model3/lstm"
    log_dir_gru="logs/model3/gru"
    file_writer1 = tf.summary.create_file_writer(log_dir_lstm)
    file_writer2 = tf.summary.create_file_writer(log_dir_gru)
    tensorboard_callback_lstm = tf.keras.callbacks.TensorBoard(log_dir=log_dir_lstm,write_images=True)
    tensorboard_callback_gru = tf.keras.callbacks.TensorBoard(log_dir=log_dir_gru,write_images=True)



    # 使用load方法载入数据
    dataset, info = tfds.load('imdb_reviews/subwords8k',data_dir='data', download=False,with_info=True,as_supervised=True)

    # 训练集测试集
    train_dataset, test_dataset = dataset['train'], dataset['test']

    # 查看vocabulary信息
    tokenizer = info.features['text'].encoder
    print('vocabulary size: ', tokenizer.vocab_size)

    ## 编码字符串
    sample_string = 'Hello word , Tensorflow'
    tokenized_string = tokenizer.encode(sample_string)
    print('tokened id: ', tokenized_string)

    ## 解码会原字符串
    src_string = tokenizer.decode(tokenized_string)
    print('original string: ', src_string)


    # 训练集shuffle：防止过拟合
    BUFFER_SIZE=64   ## BUFFER_SIZE大小
    train_dataset = train_dataset.shuffle(BUFFER_SIZE)

    # padding
    # padded_batch 参数：
    # batch_size：batch的大小
    # padded_shapes：dataset不同特征的shape，是padding具体形状的依据
    BATCH_SIZE=16     # batch_size大小
    # 训练集padding
    train_dataset = train_dataset.padded_batch(batch_size=BATCH_SIZE
                                               ,padded_shapes=tf.compat.v1.data.get_output_shapes(train_dataset))

    # 测试集padding
    test_dataset = test_dataset.padded_batch(batch_size=BATCH_SIZE
                                             ,padded_shapes=tf.compat.v1.data.get_output_shapes(test_dataset))

    # 两层双向LSTM模型
    embedding_size = 32




    # lstm 模型配置
    lstm_model = LSTM_model()
    lstm_model.compile(loss='binary_crossentropy',
                       optimizer='adam',
                       metrics=['accuracy'])

    # gru 模型配置
    gru_model = GRU_model()
    gru_model.compile(optimizer=keras.optimizers.Adam(),
                      loss=keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])


    # lstm 模型查看
    print(lstm_model.summary())

    # gru 模型查看
    print(gru_model.summary())

    # lstm 模型训练
    # 注意使用 fit训练 dataset类型数据时，不能使用validation_split参数
    # 输入参数：
    # train_dataset：训练数据集，包括输入和目标输出
    # epochs：全量样本的训练轮数
    # validation_data：验证数据集，包括输入和目标输出

    # 输出：
    # 模型每一次训练时在验证集、训练集上的metrics配置的指标,loss指标默认存在metrics中
    lstm_history = lstm_model.fit(train_dataset,epochs=5,validation_data=test_dataset,use_multiprocessing=True,callbacks=[tensorboard_callback_lstm])
    # lstm_history = lstm_model.fit_generator(train_dataset,epochs=10,validation_data=test_dataset,verbose=1)


    # 模型每一次训练时在验证集、训练集上的metrics配置的指标,loss指标默认存在metrics中
    gru_history = gru_model.fit(train_dataset,epochs=5,validation_data=test_dataset,use_multiprocessing=True,callbacks=[tensorboard_callback_gru])



    # lstm模型验证
    print(lstm_model.evaluate(test_dataset))



    # gru模型验证
    print(gru_model.evaluate(test_dataset))























