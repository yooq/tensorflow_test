import  os
import  tensorflow as tf
import  numpy as np
from    tensorflow import keras
from    tensorflow.keras import Sequential, layers
from    PIL import Image
from    matplotlib import pyplot as plt
from src.mnist_data.mnist import bendi_mnist
import tensorflow as tf
from tensorflow.keras import layers


tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')


def save_images(imgs, name):
    new_im = Image.new('L', (280, 280))

    index = 0
    for i in range(0, 280, 28):
        for j in range(0, 280, 28):
            im = imgs[index]
            im = Image.fromarray(im, mode='L')
            new_im.paste(im, (i, j))
            index += 1

    new_im.save(name)


h_dim = 20
batchsz = 512
lr = 1e-3




(x_train, y_train), (x_test, y_test) = bendi_mnist()
x_train, x_test = x_train.astype(np.float32) / 255., x_test.astype(np.float32) / 255.
# we do not need label
train_db = tf.data.Dataset.from_tensor_slices(x_train)
train_db = train_db.shuffle(batchsz * 5).batch(batchsz)
test_db = tf.data.Dataset.from_tensor_slices(x_test)
test_db = test_db.batch(batchsz)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)



class AE(keras.Model):

    def __init__(self):
        super(AE, self).__init__()

        # Encoders
        self.encoder = Sequential([
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(h_dim)
        ])

        # Decoders
        self.decoder = Sequential([
            layers.Dense(128, activation=tf.nn.relu),
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(784)
        ])


    def call(self, inputs, training=None):
        # [b, 784] => [b, 20]
        h = self.encoder(inputs)
        print(h.shape)
        # [b, 20] => [b, 784]
        x_hat = self.decoder(h)

        return h,x_hat



model = AE()
x_train = tf.reshape(x_train, [-1, 784])

x= model(x_train)  #前向传播
print(x)
model.build(input_shape=(None, 784))
model.summary()
model.compile(optimizer='adam',loss=tf.keras.losses.mse)
model.fit(x_train,x,epochs=10)
