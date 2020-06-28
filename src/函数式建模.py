import tensorflow as tf
from tensorflow import  keras
from mnist_data import mnist

(x_train, y_train), (x_test, y_test) = mnist.bendi_mnist()

(num,lens,features) = x_train.shape

input = keras.Input(shape=(lens,features))

x = keras.layers.Flatten()(input)
x = keras.layers.Dense(128,activation='relu')(x)
x = keras.layers.Dropout(0.3)(x)

output = keras.layers.Dense(10,activation='softmax')(x)

model = keras.Model(inputs=input,outputs = output)
print(model.summary())

