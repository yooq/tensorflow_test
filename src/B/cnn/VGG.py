from src.mnist_data.mnist import bendi_mnist
import tensorflow as tf
from tensorflow.keras import layers


(x_train, y_train), (x_test, y_test) = bendi_mnist()
y_train = y_train.reshape([60000,1])
print(x_train.shape)
print(y_train.shape)

def preprocess(x,y):
    '''
    数据预处理
    :param x:
    :param y:
    :return:
    '''

    x = tf.cast(x,dtype=tf.float32)/255.
    y = tf.cast(y,dtype=tf.int32)

    return x,y

train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_db = train_db.shuffle(10000).map(preprocess).batch(64)

test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_db = test_db.map(preprocess).batch(64)


# 卷积层
conv = [

    layers.Conv1D(64,kernel_size=3,padding='same',activation='relu')
    , layers.Conv1D(64,kernel_size=3,padding='same',activation='relu')
    ,layers.MaxPool1D(pool_size=2,strides=2,padding='same')

    ,layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')
    , layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')
    , layers.MaxPool1D(pool_size=4, strides=2, padding='same')

    ,layers.Conv1D(256, kernel_size=3, padding='same', activation='relu')
    , layers.Conv1D(256, kernel_size=3, padding='same', activation='relu')
    , layers.MaxPool1D(pool_size=2, strides=2, padding='same')

    ,layers.Conv1D(512, kernel_size=3, padding='same', activation='relu')
    , layers.Conv1D(512, kernel_size=3, padding='same', activation='relu')
    , layers.MaxPool1D(pool_size=2, strides=2, padding='same')

    ,layers.Conv1D(512, kernel_size=3, padding='same', activation='relu')
    , layers.Conv1D(512, kernel_size=3, padding='same', activation='relu')
    , layers.MaxPool1D(pool_size=2, strides=2, padding='same')
]




def main_net():
    '''
    [b,32,32,3] ===> [b,1,1,512]
    :return:
    '''
    conv_net = tf.keras.Sequential(conv)


    # # 全连接层
    fc_net = tf.keras.Sequential([
           layers.Dense(256,activation='relu')
           ,layers.Dense(128,activation='relu')
           ,layers.Dense(100,activation='relu')
       ])



    # 卷积层输入
    conv_net.build(input_shape=[None,28,28])

    #全连接层输入
    fc_net.build(input_shape=[None,512])

    variable =conv_net.trainable_variables+fc_net.trainable_variables
    #训练
    for epoch in range(50):
        for step,(x,y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                out =conv_net(x)
                logist = fc_net(out)
                y_onehot = tf.one_hot(y,depth=100)


                loss = tf.losses.categorical_crossentropy(y_onehot,logist,from_logits=True)
                loss =tf.reduce_mean(loss)

            grads = tape.gradient(loss,variable)
            optimizer = tf.keras.optimizers.Adam(1e-3)
            optimizer.apply_gradients(zip(grads,variable))
            if step % 100 ==0:
                print(epoch,step,'loss:  ',float(loss))


if __name__ == '__main__':
    main_net()
