import tensorflow as tf
from tensorflow.keras import layers

# 创建网络
inputs = tf.keras.Input(shape=(784,),name='img')
h1 = layers.Dense(32,activation='relu')(inputs)
h2 = layers.Dense(32,activation='relu')(h1)
outputs = layers.Dense(10,activation='softmax')(h2)

model = tf.keras.Model(inputs=inputs,outputs=outputs,name='mnist_model')

model.summary()


# 训练及测试
(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000,784).astype('float32')/255.
x_test = x_test.reshape(10000,784).astype('float32')/255.
model.compile(
                optimizer=tf.keras.optimizers.RMSprop(),
                loss = tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy']
              )


# 回调函数，可以提前停止函数，或者在指定情况进行指定操作
# callbacks =  tf.keras.callbacks.EarlyStopping(
#         monitor='accuracy',
#         min_delta = 0.9508,
#         patience=3,
#         verbose=1
#         )


# 前十组为0.001 后面指数减少,学习率
def scheduler(epoch):
  if epoch < 3:
    return 0.01
  else:
    return 0.001 * tf.math.exp(0.1 * (10 - epoch))

learningreatescheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)


callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir='logdir'), #log_dir将输出的日志保存在所要保存的路径中
    tf.keras.callbacks.ModelCheckpoint('output_model_file', save_best_only = True), #save_best_only=False ，#保存所有模型    # save_weights_only=True ，#仅保留权重参数
    tf.keras.callbacks.EarlyStopping(patience=2, min_delta=0.2),
    learningreatescheduler
]


# 回调函数可以自定义
# class MyCallback(tf.keras.callbacks.Callback):
#     def __init__(self):
#         super(MyCallback,self).__init__()
#
#     def on_epoch_stop(self,batch,logs):
#         if (logs.get('accuracy') < 0.9015):
#             print("\nReached 60% accuracy so cancelling training!")
#             self.model.stop_training = False
# callbacks = MyCallback()

history = model.fit(x_train,y_train
                    ,callbacks=callbacks
                    # ,callbacks=[callbacks,learningreatescheduler]
                    ,batch_size=64
                    ,epochs=5
                    ,validation_split=0.2
                    )
print(history.history)
