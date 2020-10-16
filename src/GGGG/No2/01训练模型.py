import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

inputs = keras.Input(shape=(784,),name='mnist_input')
h1 = layers.Dense(64,activation='relu')(inputs)
h1 = layers.Dense(64,activation='relu')(h1)
outputs = layers.Dense(10,activation='softmax')(h1)

model = keras.Model(inputs,outputs)
model.compile(
            optimizer=keras.optimizers.RMSprop(),
            loss = keras.losses.SparseCategoricalCrossentropy(),
            metrics=[keras.metrics.SparseCategoricalAccuracy()]
              )

(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000,784).astype('float32')/255.
x_test = x_test.reshape(10000,784).astype('float32')/255.

x_val = x_train[-10000:]
y_val = y_train[-10000:]

x_train = x_train[:-10000]
y_train = y_train[:-10000]

history = model.fit(x_train,y_train,batch_size=64,epochs=3,validation_data=(x_val,y_val))
print('history:')
'''
{
'loss': [0.34224696431159973, 0.1598119018816948, 0.11419392880886793], 
'sparse_categorical_accuracy': [0.90258, 0.95176, 0.9655],
'val_loss': [0.20992365552186967, 0.13167334056198596, 0.11136388760656118],
'val_sparse_categorical_accuracy': [0.936, 0.9618, 0.9685]
}
'''
print(history.history)

result = model.evaluate(x_test, y_test, batch_size=128)
print('evaluate:')
print(result)
pred = model.predict(x_test[:2])
print('predict:')
print(pred)
