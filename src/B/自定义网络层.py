from tensorflow.keras import layers
from  tensorflow import keras
import tensorflow as tf

class MyDense(layers.Layer):
    def __init__(self,inp_dim,outp_dim):
        super(MyDense,self).__init__()

        self.kernel = self.add_variable('w',[inp_dim,outp_dim])
        self.bias = self.add_variable('b',[outp_dim])

    def call(self,inputs,training=None):
        out = inputs @ self.kernel + self.bias
        return out


class Mymodel(keras.Model):
    def __init__(self):
        super(Mymodel, self).__init__()
        self.fc1 = MyDense(28*28,256)
        self.fc2 = MyDense(256,128)
        self.fc3 = MyDense(128,64)
        self.fc4 = MyDense(64,32)
        self.fc5 = MyDense(32,10)

    def call(self,inputs,training=None):
        x = self.fc1(inputs)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        x = tf.nn.relu(x)
        x = self.fc3(x)
        x = tf.nn.relu(x)
        x = self.fc4(x)
        x = tf.nn.relu(x)
        x = self.fc5(x)
        return x


network = Mymodel()

network.compile(
                 optimizer=tf.keras.optimizers.Adam(0.01)
                ,loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True)
                ,metrics=['acc']
               )

network.fit(shujumeiyou , epochs=,.......)
