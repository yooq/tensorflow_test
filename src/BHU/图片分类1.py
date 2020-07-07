from src.mnist_data import mnist
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow import keras

import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = mnist.bendi_mnist()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0

test_images = test_images / 255.0
plt.figure(figsize=(10,10))
for i in range(25,50):
    plt.subplot(5,5,i+1-25)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])

    plt.xlabel(class_names[train_labels[i]])
plt.show()


model = keras.Sequential(
[
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])


model.fit(train_images, train_labels, epochs=5,shuffle=10000)
print(model.evaluate(test_images, test_labels,return_dict=True))
