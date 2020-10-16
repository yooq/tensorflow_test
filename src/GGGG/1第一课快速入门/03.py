import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

encode_input = keras.Input(shape=(28,28,1),name = 'img')
h1 = layers.Conv2D(16,3,activation='relu')(encode_input)
h1 = layers.Conv2D(32,kernel_size=3,activation='relu')(h1)
h1 = layers.MaxPool2D(strides=3)(h1)
h1 = layers.Conv2D(32,3,activation='relu')(h1)
h1 = layers.Conv2D(16,3,activation='relu')(h1)
encode_output = layers.GlobalMaxPool2D()(h1)

encode_model = keras.Model(inputs=encode_input,outputs=encode_output,name='encoder')
encode_model.summary()



h2 = layers.Reshape((4,4,1))(encode_output)
h2 = layers.Conv2DTranspose(16,3,activation='relu')(h2)  #卷积的逆过程
h2 = layers.Conv2DTranspose(32,3,activation='relu')(h2)
h2 = layers.UpSampling2D(3)(h2)
h2 = layers.Conv2DTranspose(16,3,activation='relu')(h2)
decode_output = layers.Conv2DTranspose(1,3,activation='relu')(h2)

autoencoder = keras.Model(inputs=encode_input, outputs=decode_output, name='autoencoder')
autoencoder.summary()
