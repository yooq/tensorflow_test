import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

class MyLayer(layers.Layer):
    def __init__(self, unit=32):
        super(MyLayer, self).__init__()
        self.unit = unit

    def build(self, input_shape):
        self.weight = self.add_weight(shape=(input_shape[-1], self.unit),
                                      initializer=keras.initializers.RandomNormal(),
                                      trainable=True)  #trainable 如果为false则表明该权重不可训练，不会更新
        self.bias = self.add_weight(shape=(self.unit,),
                                    initializer=keras.initializers.Zeros(),
                                    trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.weight) + self.bias


my_layer = MyLayer(unit=3)

x = tf.ones((3, 5))
out = my_layer(x)
print(out)
my_layer = MyLayer(3)

x = tf.ones((2, 2))
out = my_layer(x)
print(out)

