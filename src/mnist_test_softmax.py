from mnist_data import mnist
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.bendi_mnist()

model = tf.keras.Sequential()

model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))
model.compile(optimizer='adam',
              loss=tf.keras.losses.sparse_categorical_crossentropy,metrics=['acc']
              )
model.fit(x_train,y_train,epochs=5,verbose=1)

model.evaluate(x_test,y_test)





