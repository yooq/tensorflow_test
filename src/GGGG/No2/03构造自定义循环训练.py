import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

# 载入数据
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') /255
x_test = x_test.reshape(10000, 784).astype('float32') /255


inputs = keras.Input(shape=(784,),name = 'dignits')
x = layers.Dense(64,activation='relu',name='dense_1')(inputs)
x = layers.Dense(64,activation='relu',name='dense_2')(x)
outputs = layers.Dense(10,activation='softmax',name='predictions')(x)
model = keras.Model(inputs=inputs,outputs=outputs)
optimizer = keras.optimizers.SGD(learning_rate=1e-3)
loss_fn = keras.losses.SparseCategoricalCrossentropy()

batch_size = 64
train_dateset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_dateset = train_dateset.shuffle(buffer_size=1024).batch(batch_size)

train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

for epoch in range(3):
    print('epoch: ',epoch)
    for step,(x_batch_train,y_batch_train) in enumerate(train_dateset):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train)
            loss_value = loss_fn(y_batch_train,logits)
        grads = tape.gradient(loss_value,model.trainable_variables)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))

        train_acc_metric(y_batch_train,logits)

        if step%200==0:
            print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
            print('Seen so far: %s samples' % ((step + 1) * 64))

        train_acc = train_acc_metric.result()
        print('Training acc over epoch: %s' % (float(train_acc),))

        # 重置统计参数
        train_acc_metric.reset_states()
