import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
# from IPython.display import SVG

'''

UpSampling2D可以看作是Pooling的反向操作，就是采用Nearest Neighbor interpolation来进行放大，
说白了就是复制行和列的数据来扩充feature map的大小。反向梯度传播的时候，应该就是每个单元格的梯度的和（猜测）。
Conv2DTranspose就是正常卷积的反向操作，无需多讲。

'''

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape((-1, 28*28)) / 255.0
x_test = x_test.reshape((-1, 28*28)) / 255.0

# (60000, 784)   (60000,)
# ---------
# (10000, 784)   (10000,)
print(x_train.shape, ' ', y_train.shape)
print('---------')
print(x_test.shape, ' ', y_test.shape)

code_dim = 32
inputs = layers.Input(shape=(x_train.shape[1],),name='inputs')
code = layers.Dense(code_dim,activation='relu',name='code')(inputs)
outputs = layers.Dense(x_train.shape[1],activation='softmax',name='outputs')(code)

auto_encoder = keras.Model(inputs, outputs)
auto_encoder.summary()

encoder = keras.Model(inputs,code)


layers.UpSampling2D(
decoder_input = keras.Input((code_dim,))
decoder_output = auto_encoder.layers[-1](decoder_input)
decoder = keras.Model(decoder_input, decoder_output)

auto_encoder.compile(optimizer='adam',
                    loss='binary_crossentropy')
