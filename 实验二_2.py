import tensorflow as tf
import datetime
from tensorboard.plugins.hparams import api as hp


def create_model(log_dir,x_train,y_train,hparams):
  model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]),
                    filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid',
                    activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
    tf.keras.layers.Dense(10, activation='softmax')
  ])

  model.compile(optimizer=hparams[HP_OPTIMIZER],
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  model.fit(x=x_train, y=y_train,
                        batch_size=hparams[HP_BATCH_SIZE],
                        epochs=hparams[HP_EPOCHS], validation_split=hparams[HP_VALI_SPLIT],
                        callbacks=[tf.keras.callbacks.TensorBoard(log_dir),  # log metrics
        hp.KerasCallback(log_dir, hparams),])
  return model


def run(run_dir, x_train,y_train,x_test,y_test,hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    model = create_model(run_dir,x_train,y_train,hparams)
    _, accuracy = model.evaluate(x_test, y_test)
    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)


if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data('mnist.npz')
    # x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))

    HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam','Adagrad']))
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([50,100]))
    HP_EPOCHS = hp.HParam('epochs', hp.Discrete([50,100]))
    HP_VALI_SPLIT = hp.HParam('validation_split', hp.Discrete([0.1]))
    METRIC_ACCURACY = 'accuracy'

    # HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1,0.2))
    # HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam']))
    # HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([50]))
    # HP_EPOCHS = hp.HParam('epochs', hp.Discrete([10]))
    # HP_VALI_SPLIT = hp.HParam('validation_split', hp.Discrete([0.1]))
    # METRIC_ACCURACY = 'accuracy'

    with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
        tf.summary.image("25 training data examples", x_train[0:25], max_outputs=25, step=0)
        hp.hparams_config(
            hparams=[HP_DROPOUT, HP_OPTIMIZER,HP_BATCH_SIZE,HP_EPOCHS,HP_VALI_SPLIT],
            metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
        )


    # log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # file_writer = tf.summary.create_file_writer(log_dir)
    #
    # with file_writer.as_default():
    #     tf.summary.image("25 training data examples", x_train[0:25], max_outputs=25, step=0)

    session_num = 0

    for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
        for optimizer in HP_OPTIMIZER.domain.values:
            for batch in HP_BATCH_SIZE.domain.values:
                for epoch in HP_EPOCHS.domain.values:
                    for vds in HP_VALI_SPLIT.domain.values:
                        hparams = {
                            HP_DROPOUT: dropout_rate,
                            HP_OPTIMIZER: optimizer,
                            HP_BATCH_SIZE: batch,
                            HP_EPOCHS: epoch,
                            HP_VALI_SPLIT: vds,
                        }
                        run_name = "run-%d" % session_num
                        print('--- Starting trial: %s' % run_name)
                        print({h.name: hparams[h] for h in hparams})
                        run('logs/hparam_tuning/' + run_name, x_train,y_train,x_test,y_test,hparams)
                        session_num += 1