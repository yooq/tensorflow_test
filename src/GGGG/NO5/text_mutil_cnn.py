import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.preprocessing.sequence import pad_sequences

num_features = 3000
sequence_length = 300
embedding_dimension = 100
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=num_features)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


x_train = pad_sequences(x_train, maxlen=sequence_length)
x_test = pad_sequences(x_test, maxlen=sequence_length)
# print(x_train)
print(x_train.shape)


filter_size = [3,4,5]
def convolution():
    inn = layers.Input(shape=(sequence_length,embedding_dimension,1))
    cnns = []
    for size in filter_size:
        conv = layers.Conv2D(filters=64,kernel_size=(size,embedding_dimension)
                             ,strides=1,padding='valid',activation='relu')(inn)
        pool = layers.MaxPool2D(pool_size=(sequence_length-size+1,1)
                                ,padding='valid')(conv)
        print(pool.shape)
        print(pool)
        print('-----------------------')
        cnns.append(pool)
    print(cnns)

    outt = layers.concatenate(cnns)
    print(outt.shape)
    model = keras.Model(inputs = inn,outputs=outt)
    return model

def cnn_mulfilter():
    '''
        num_features = 3000
        sequence_length = 300
        embedding_dimension = 100
    :return:
    '''
    model = keras.Sequential([
        layers.Embedding(input_dim=num_features,output_dim=embedding_dimension,input_length=sequence_length),
        layers.Reshape((sequence_length,embedding_dimension,1)), #(300,100,1)
        convolution(),
        layers.Flatten(),
        layers.Dense(10,activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1,activation='sigmoid')
    ])

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    return model

model = cnn_mulfilter()
model.summary()
tensorboard_callback = callbacks.TensorBoard(log_dir='log_dir', histogram_freq=1, write_images=True)

history = model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.1,callbacks=[tensorboard_callback])
