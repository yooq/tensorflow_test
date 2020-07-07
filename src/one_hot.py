from mnist_data import mnist
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.bendi_mnist()

'''
将标签转成one-hot
'''
# 方法一
y_train_keras = to_categorical(y_train)
print(y_train_keras[0:3])
y_test_keras = to_categorical(y_test)
print(y_test_keras[0:3])

# 方法二
# import numpy as np
# l = len(y_train)
# y_train_numpy = np.eye(l)[y_test]
# print(y_train_numpy[0:3])

# 方法三
# from sklearn.preprocessing import OneHotEncoder
#
# one_hot = OneHotEncoder(n_values='auto')
# y_train_sklearn = one_hot.fit_transform(y_train.reshape(len(y_train),1))
# print(y_train_sklearn) #这返回值是个稀疏型
# print(y_train_sklearn[0:3].toarray())


model = tf.keras.Sequential()

model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128,activation='relu'
                                # ,kernel_regularizer='l2'  #正则化
                                ,kernel_regularizer=tf.keras.regularizers.l2(0.001)
                                ))
model.add(tf.keras.layers.Dropout(rate=0.1))
model.add(tf.keras.layers.Dense(256,activation='relu'))
model.add(tf.keras.layers.Dense(256,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001,momentum=0.9),
              loss=tf.keras.losses.categorical_crossentropy,metrics=['acc']  #one-hot标签与数字标签使用不同的损失函数
              )
model.fit(x_train,y_train_keras
          ,epochs=5
          ,verbose=1
          ,validation_data=(x_test,y_test_keras) #验证数据集

          )
model
# model.evaluate(x_test,y_test_keras)
