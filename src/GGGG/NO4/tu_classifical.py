import tensorflow as tf
from  tensorflow.keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt


(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

print(train_labels[0:10])
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

train_data = tf.data.Dataset.from_tensor_slices((train_images,train_labels))
train_data = train_data.shuffle(3)
train_data = train_data.batch(32)
train_data = train_data.repeat(1)
print('=======================')
print(train_data)

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.colorbar()
#     plt.grid(False)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()


model = keras.Sequential(
[
    layers.Flatten(input_shape=[28, 28]),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

model.fit(train_data,epochs=10)

import numpy as np
print(np.shape(test_images))
print(np.shape([test_images[0]]))
x_test = np.reshape(test_images[0],newshape=(1,28,28))
predictions = model.predict(x_test)
print(predictions)
print(np.argmax(predictions))
print(test_labels[0])
