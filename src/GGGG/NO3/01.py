import tensorflow as tf
tf.keras.backend.clear_session()
from tensorflow.keras import layers
from tensorflow import keras

# class MyLayer(tf.keras.Model):
#     def __init__(self,input_dim=32,unit=32):
#         super(MyLayer,self).__init__()
#
#         w_init = tf.random_normal_initializer()
#         self.weight = tf.Variable(initial_value=w_init(
#             shape=(input_dim, unit), dtype=tf.float32), trainable=True)
#
#         b_init = tf.zeros_initializer()
#         self.bias = tf.Variable(initial_value=b_init(
#             shape=(unit,), dtype=tf.float32), trainable=True)
#
#     def call(self,inputs):
#         return tf.matmul(inputs,self.weight)+self.bias


class MyLayer(layers.Layer):
    def __init__(self, input_dim=32, unit=32):
        super(MyLayer, self).__init__()
        self.weight = self.add_weight(shape=(input_dim, unit),
                                      initializer=keras.initializers.RandomNormal(),
                                      trainable=True)  #trainable 表示是否可以训练
        self.bias = self.add_weight(shape=(unit,),
                                    initializer=keras.initializers.Zeros(),
                                    trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.weight) + self.bias

x = tf.ones((3,5))
my_layer = MyLayer(5,4)
out = my_layer(x)
print(out)
