from sklearn.model_selection import  train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle


def generate(sample_size, mean, cov, diff, regression):
    num_classes = 2  # len(diff)
    samples_per_class = int(sample_size / 2)

    X0 = np.random.multivariate_normal(mean, cov, samples_per_class)
    Y0 = np.zeros(samples_per_class)

    for ci, d in enumerate(diff):
        X1 = np.random.multivariate_normal(mean + d, cov, samples_per_class)
        Y1 = (ci + 1) * np.ones(samples_per_class)

        X0 = np.concatenate((X0, X1))
        Y = np.concatenate((Y0, Y1))

    if regression == False:  # one-hot  0 into the vector "1 0
        class_ind = [Y == class_number for class_number in range(num_classes)]
        Y = np.asarray(np.hstack(class_ind), dtype=np.float32)
    X, Y = shuffle(X0, Y)

    return X, Y

np.random.seed(10)
num_classes =2
mean = np.random.randn(num_classes)
cov = np.eye(num_classes)
X_train, Y_train = generate(1000, mean, cov, [3.0],True)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(10,input_shape=(2,),activation='relu'))
model.add( tf.keras.layers.Dense(100,activation='relu'))
model.add(tf.keras.layers.Dense(50,activation='relu'))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

model.compile(optimizer=tf.keras.optimizers.Adam(0.001)
              ,loss=tf.keras.losses.binary_crossentropy
              ,metrics=['acc']

              )
history = model.fit(X_train,Y_train
                    ,epochs=200
                    ,batch_size=20,shuffle=100
                    # ,validation_data= 测试数据,可做交叉验证
                    # ,validation_freq=2  没两次epoch,放一次测试数据。  训练数据，训练数据，测试数据，训练数据。。。
                    # 测试也可以换一种方式，下面
                    )

model.save_weights('weights.ckpt')  #保留参数，不保留模型

del model #s删除模型

# 重新定义原模型结构
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(10,input_shape=(2,),activation='relu'))
model.add( tf.keras.layers.Dense(100,activation='relu'))
model.add(tf.keras.layers.Dense(50,activation='relu'))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

model.compile(optimizer=tf.keras.optimizers.Adam(0.001)
              ,loss=tf.keras.losses.binary_crossentropy
              ,metrics=['acc'])


model.load_weights('weights.ckpt') #加载参数后不用训练
print(model.trainable_variables)



# 讲模型整个结构及参数全部保存

model.save()
model = tf.keras.models.load_model()  #即可


# 将模型工业化导出，可供其他模型使用
tf.saved_model.save()
tf.saved_model.load()
