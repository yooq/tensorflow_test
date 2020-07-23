import tensorflow as tf
from tensorflow.keras import layers,Sequential

'''
自定义模型结构
'''

class MyModel(tf.keras.Model):
    def __init__(self,num_classes=10):
        super(MyModel,self).__init__(name='mymodel')
        self.num_classes = num_classes
        self.layer1 = layers.Dense(32,activation='relu')
        self.layer2 = layers.Dense(num_classes,activation='softmax')

    def call(self,inputs):
        h1 = self.layer1(inputs)
        h2 = self.layer2(h1)
        return h2

    def compute_out_shape(self,input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        print('shape:------>', shape)
        return tf.TensorShape(shape)



import numpy as np

train_x = np.random.random((1000, 72))
train_y = np.random.random((1000, 10))

model = MyModel(num_classes=10)
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
             loss=tf.keras.losses.categorical_crossentropy,
             metrics=['accuracy'])

model.fit(train_x, train_y, batch_size=16, epochs=5)
sha = model.compute_out_shape(input_shape=(16,))
print(sha)
