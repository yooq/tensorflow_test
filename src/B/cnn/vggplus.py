from src.mnist_data.mnist import bendi_mnist
import tensorflow as tf
from tensorflow.keras import layers


(x_train, y_train), (x_test, y_test) = bendi_mnist()
x_train =x_train/255.
print(x_train.shape)
print(y_train.shape)

def preprocess(x,y):
    '''
    数据预处理
    :param x:
    :param y:
    :return:
    '''

    x = tf.cast(x,dtype=tf.float32)/255.
    y = tf.cast(y,dtype=tf.int32)

    return x,y


model = tf.keras.Sequential()

model.add(layers.Conv1D(64,kernel_size=3,input_shape=[28,28],strides=1,padding='same',activation='relu'))
model.add(layers.Conv1D(64,kernel_size=3,strides=1,padding='same',activation='relu'))
model.add(layers.MaxPool1D(pool_size=2,strides=2,padding='same'))

model.add(layers.Conv1D(128,kernel_size=3,strides=1,padding='same',activation='relu'))
model.add(layers.Conv1D(128,kernel_size=3,strides=1,padding='same',activation='relu'))
model.add(layers.MaxPool1D(pool_size=2,strides=2,padding='same'))

model.add(layers.Conv1D(256,kernel_size=3,strides=1,padding='same',activation='relu'))
model.add(layers.Conv1D(256,kernel_size=3,strides=1,padding='same',activation='relu'))
model.add(layers.MaxPool1D(pool_size=2,strides=2,padding='same'))

model.add(layers.Conv1D(512,kernel_size=3,strides=1,padding='same',activation='relu'))
model.add(layers.Conv1D(512,kernel_size=3,strides=1,padding='same',activation='relu'))
model.add(layers.MaxPool1D(pool_size=2,strides=2,padding='same'))

model.add(layers.Conv1D(512,kernel_size=3,strides=1,padding='same',activation='relu'))
model.add(layers.Conv1D(512,kernel_size=3,strides=1,padding='same',activation='relu'))
model.add(layers.MaxPool1D(pool_size=2,strides=2,padding='same'))


model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(100,activation='relu'))
model.summary()

model.compile(
                optimizer=tf.optimizers.Adam(1e-3)
              ,loss = tf.losses.mse
              ,metrics=['acc']
)


model.fit(x_train,y_train,epochs=100)
