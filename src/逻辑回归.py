from sklearn.datasets import  load_wine
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
history = model.fit(X_train,Y_train,epochs=200,batch_size=20)

acc =history.history['acc']
loss = history.history['loss']


print(history.history.keys())  #可以查看 dict_keys(['loss', 'acc'])

plt.figure()
plt.plot([i for i in range(200)],acc,label='acc')
# plt.plot([i for i in range(200)],loss,label='loss')

plt.plot(history.epoch,history.history.get('loss'))
plt.legend()
plt.show()

