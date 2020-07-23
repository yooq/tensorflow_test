import tensorflow as tf
from tensorflow.keras import layers

'''
简单模型
'''
model = tf.keras.Sequential()
model.add(layers.Dense(32,input_shape=(72,),activation='relu'))
model.add(layers.Dense(32, activation='relu'))


'''
网络配置
'''
# model.add(layers.Dense(32, activation='sigmoid'))
# model.add(layers.Dense(32, activation=tf.sigmoid))
# model.add(layers.Dense(32, kernel_initializer='orthogonal'))  #层的初始化方案 默认Glorot uniform
# model.add(layers.Dense(32, kernel_initializer=tf.keras.initializers.glorot_normal))   #层的初始化方案 默认Glorot uniform
# model.add(layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.01)))   #二范式
# model.add(layers.Dense(32, kernel_regularizer=tf.keras.regularizers.l1(0.01)))   #一范式

model.add(layers.Dense(10, activation='softmax'))

'''
优化，训练和评估

'''
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
             loss=tf.keras.losses.categorical_crossentropy,
             metrics=[tf.keras.metrics.categorical_accuracy])

'''
训练
'''
import numpy as np

train_x = np.random.random((1000, 72))
train_y = np.random.random((1000, 10))

val_x = np.random.random((200, 72))
val_y = np.random.random((200, 10))
#
# model.fit(train_x, train_y, epochs=10, batch_size=100,
#           validation_data=(val_x, val_y))


'''
训练
'''
dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)) #把给定的元组、列表和张量等数据进行特征切片
dataset = dataset.batch(32) #批次
dataset = dataset.shuffle(1000).repeat()  #洗牌和打乱
val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y))
val_dataset = val_dataset.batch(32)
val_dataset = val_dataset.repeat()

model.fit(dataset, epochs=10, steps_per_epoch=10,
          validation_data=val_dataset, validation_steps=3)  # steps_per_epoch 一个epoch包含的步数（每一步是一个batch的数据送入）
# 个人理解这就是个交叉验证思路，上面把数据分为32个批次，每一个epochs选10个批次来训练

print('++++'*20)
test_x = np.random.random((1000, 72))
test_y = np.random.random((1000, 10))
eva1 = model.evaluate(test_x, test_y, batch_size=32)
print(eva1)
test_data = tf.data.Dataset.from_tensor_slices((test_x, test_y))
test_data = test_data.batch(32).repeat()
eva2 = model.evaluate(test_data, steps=30)
print(eva2)
# predict
result = model.predict(test_x, batch_size=32)
print(result)
