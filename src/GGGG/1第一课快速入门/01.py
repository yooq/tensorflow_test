import tensorflow as tf
from tensorflow.keras import layers
import logging
logging.basicConfig(level=logging.ERROR)
print(tf.__version__)
print(tf.keras.__version__)


# 堆叠模型
model = tf.keras.Sequential()
model.add(layers.Dense(10,input_shape=(72,),activation='relu'))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
model.summary()

# 网络配置，包括正则，BN等
model.add(layers.Dense(15,kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model.add(layers.Dense(10,kernel_initializer=tf.keras.initializers.glorot_normal))
model.summary()

# 设置训练流程
model.compile(
                optimizer=tf.keras.optimizers.Adam(0.001),
                loss = tf.keras.losses.categorical_crossentropy,
                metrics=[tf.keras.metrics.categorical_accuracy]
              )

# 准备数据
import numpy as np
train_x = np.random.random((1000,72))
trian_y = np.random.random((1000,10))

val_x = np.random.random((200,72))
val_y = np.random.random((200,10))
# model.fit(train_x,trian_y,epochs=10,batch_size=100,validation_data=(val_x,val_y))

# 利用tf.data输入数据
dataset = tf.data.Dataset.from_tensor_slices((train_x,trian_y)) #要求第一维度必须一样
# print(dataset) #<TensorSliceDataset shapes: ((72,), (10,)), types: (tf.float64, tf.float64)>
dataset = dataset.batch(32)
dataset = dataset.repeat()

# print(dataset)  # <<RepeatDataset shapes: ((None, 72), (None, 10)), types: (tf.float64, tf.float64)>

val_dataset = tf.data.Dataset.from_tensor_slices((val_x,val_y))
val_dataset = val_dataset.batch(32)
val_dataset = val_dataset.repeat()
model.fit(dataset,epochs=10,steps_per_epoch=30,validation_data=val_dataset,validation_steps=3)


print('++++++++++++++++++++++++++++++++')
# 模型评估
test_x = np.random.random((1000, 72))
test_y = np.random.random((1000, 10))
model.evaluate(test_x, test_y, batch_size=32)
test_data = tf.data.Dataset.from_tensor_slices((test_x, test_y))
test_data = test_data.batch(32).repeat()
model.evaluate(test_data, steps=30)  #输入数据和标签,输出损失和精确度.
# predict
result = model.predict(test_x, batch_size=32) #输入测试数据,输出预测结果
print(result)


# api函数式模型
print('# api函数式模型')
input_x = tf.keras.Input(shape=(72,))
hidden1 = layers.Dense(32,activation='relu')(input_x)
hidden2 = layers.Dense(32,activation='relu')(hidden1)
hidden3 = layers.Dense(32,activation='relu')(hidden2)
pred = layers.Dense(10,activation='softmax')(hidden3)
model = tf.keras.Model(inputs=input_x,outputs=pred)
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss = tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy']
              )
model.fit(train_x,trian_y,batch_size=32,epochs=5)


# 模块子类化
print('# 模块子类化')
class MyModel(tf.keras.Model):
    def __init__(self,num_classes=10):
        super(MyModel,self).__init__(name='MyModel')
        self.num_classes = num_classes
        self.layer1 = layers.Dense(32,activation='relu')
        self.layer2 = layers.Dense(num_classes,activation='softmax')

    def call(self,inputs):
        h1 = self.layer1(inputs)
        out = self.layer2(h1)
        return out

    def compute_output_shape(self,input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)

model = MyModel(num_classes =10)
model.compile(
                optimizer=tf.keras.optimizers.RMSprop(0.001),
                loss = tf.keras.losses.categorical_crossentropy,
                metrics=['accuracy']
              )
model.fit(train_x,trian_y,batch_size=16,epochs=5)
