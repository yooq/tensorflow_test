import  tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

(x_train,y_train),(x_test,y_test) = keras.datasets.boston_housing.load_data()

# (404, 13)   (404,)
# (102, 13)   (102,)
print(x_train.shape, ' ', y_train.shape)
print(x_test.shape, ' ', y_test.shape)

model = keras.Sequential([
    layers.Dense(32,activation='sigmoid',input_shape=(13,)),
    layers.BatchNormalization(),
    layers.Dense(32,activation='sigmoid'),
    layers.BatchNormalization(),
    layers.Dense(32,activation='sigmoid'),
    layers.Dense(1)]
)

model.compile(optimizer=keras.optimizers.SGD(0.1),
              loss = 'mean_squared_error',
              metrics=['mse']
              )

model.summary()

model.fit(x_train,y_train,epochs=10,batch_size=10,validation_data=(x_test,y_test))

print(model.metrics_names)
