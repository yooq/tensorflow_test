import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

num_words = 30000
maxlen = 200
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=num_words)
print(x_train.shape, ' ', y_train.shape)
print(x_test.shape, ' ', y_test.shape)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen, padding='post')
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen, padding='post')
print(x_train.shape, ' ', y_train.shape)
print(x_test.shape, ' ', y_test.shape)
print('----------------------------------------------')
# LSTM
def lstm_model():
    model = keras.Sequential([
        layers.Embedding(input_dim=30000,output_dim=32,input_length=maxlen),
        # return_sequences：控制hidden_state,True 输出全部，False输出最后一个
        # return_state：控制cell_state，True输出，False不输出
        layers.LSTM(32,return_sequences=True),
        layers.LSTM(1,activation='sigmoid',return_sequences=False)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    return model

# model = lstm_model()
# model.summary()
# history = model.fit(x_train, y_train, batch_size=64, epochs=5,validation_split=0.1)
# print('----------------------------------------------')


# GRU
def gru_model():
    model = keras.Sequential([
        layers.Embedding(input_dim=30000, output_dim=32, input_length=maxlen),
        # return_sequences：控制hidden_state,True 输出全部，False输出最后一个
        # return_state：控制cell_state，True输出，False不输出
        layers.GRU(32, return_sequences=True),
        layers.GRU(1, activation='sigmoid', return_sequences=False)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    return model

model1 = gru_model()
model1.summary()
history1 = model1.fit(x_train, y_train, batch_size=64, epochs=5,validation_split=0.1)
print(history1.history['accuracy'])  #[0.5742667, 0.7768, 0.84826666, 0.7300444, 0.8373778]
