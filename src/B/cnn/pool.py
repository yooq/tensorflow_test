import tensorflow as tf
from tensorflow.keras import layers
from src.mnist_data.mnist import bendi_mnist
import  matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = bendi_mnist()
# x  shape   [1,14,14,4]
x = tf.random.normal([1,14,14,4])
pool = layers.MaxPool2D(2,strides=2)
out = pool(x) #[1,7,7,4]

print(out.shape)
# 方法2
# input, ksize, strides, padding, data_format="NHWC", name=None
out =tf.nn.max_pool2d(x,ksize=2,strides=2,padding='VALID')
print(out.shape)

# 放大图片，上采样

plt.imshow(x_train[0])
plt.show()
print(x_train[0].shape)

a = x_train[0].reshape([1,28,28,1])
print(a.shape)
layer = layers.UpSampling2D(size=5)
out=layer(a)
print(out.shape)
m = out.numpy().reshape([140,140])
plt.imshow(m)
plt.show()


