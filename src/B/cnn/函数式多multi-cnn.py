from src.mnist_data.mnist import bendi_mnist
import tensorflow as tf
from tensorflow.keras import layers


(x_train, y_train), (x_test, y_test) = bendi_mnist()
x_train =x_train/255.
print(x_train.shape)
print(y_train.shape)

input = tf.keras.Input(shape=[28,28])
x1 = layers.Conv1D(64,kernel_size=3,padding='same',activation='relu',strides=1)(input)
x2 = layers.Conv1D(64,kernel_size=1,padding='same',activation='relu',strides=1)(input)
x3 = layers.Conv1D(64,kernel_size=5,padding='same',activation='relu',strides=1)(input)
print(x3.shape)
x = tf.stack([x1,x2,x3],axis=1)
print(x.shape)

x = layers.Conv2D(6,kernel_size=[5,5],padding='same',activation='relu',strides=1)(x)
print(x.shape)
x = layers.MaxPool2D(pool_size=2,strides=2,padding='same')(x)
print(x.shape)
x= layers.Flatten()(x)
print(x.shape)
out = layers.Dense(168,activation='relu')(x)
print(out.shape)
out = layers.Dense(100,activation='relu')(out)
print(out.shape)


model =tf.keras.Model(inputs=input,outputs=x)
model.summary()
