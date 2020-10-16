import tensorflow as tf
from tensorflow.keras import layers

# 创建网络
inputs = tf.keras.Input(shape=(784,),name='img')
h1 = layers.Dense(32,activation='relu')(inputs)
h2 = layers.Dense(32,activation='relu')(h1)
outputs = layers.Dense(10,activation='softmax')(h2)

model = tf.keras.Model(inputs=inputs,outputs=outputs,name='mnist_model')

model.summary()


# 训练及测试
(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000,784).astype('float32')/255.
x_test = x_test.reshape(10000,784).astype('float32')/255.
model.compile(
                optimizer=tf.keras.optimizers.RMSprop(),
                loss = tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy']
              )
history = model.fit(x_train,y_train,batch_size=64,epochs=5,validation_split=0.2)
test_scores = model.evaluate(x_test,y_test,verbose=0)
print('test loss :',test_scores[0])
print('test acc :',test_scores[1])

# 模型保存和序列化
model.save('model_save.h5')
del model
model = tf.keras.models.load_model('model_save.h5')
