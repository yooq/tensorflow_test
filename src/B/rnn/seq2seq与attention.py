'''
这个方式不好啊

'''

from numpy import array
from numpy import argmax
from numpy import array_equal
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
import numpy as np



# 随机产生在(1,n_features)区间的整数序列，序列长度为n_steps_in
def generate_sequence(length, n_unique):
    return [np.random.randint(1, n_unique - 1) for _ in range(length)]


# 构造LSTM模型输入需要的训练数据
def get_dataset(n_in, n_out, cardinality, n_samples):
    X1, X2, y = list(), list(), list()
    for _ in range(n_samples):
        # 生成输入序列
        source = generate_sequence(n_in, cardinality)
        # 定义目标序列，这里就是输入序列的前三个数据
        target = source[:n_out]
        target.reverse()
        # 向前偏移一个时间步目标序列
        target_in = [0] + target[:-1]
        # 直接使用to_categorical函数进行on_hot编码
        src_encoded = to_categorical(source, num_classes=cardinality)
        tar_encoded = to_categorical(target, num_classes=cardinality)
        tar2_encoded = to_categorical(target_in, num_classes=cardinality)

        X1.append(src_encoded)
        X2.append(tar2_encoded)
        y.append(tar_encoded)
    return array(X1), array(X2), array(y)


# 构造Seq2Seq训练模型model, 以及进行新序列预测时需要的的Encoder模型:encoder_model 与Decoder模型:decoder_model
def define_models(n_input, n_output, n_units):

    # 训练模型中的encoder
    encoder_inputs = Input(shape=(None, n_input))
    encoder = LSTM(n_units, return_state=True)  # 实例化
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]  # 仅保留编码状态向量

    # 训练模型中的decoder
    decoder_inputs = Input(shape=(None, n_output))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)  #######################################

    # 新序列预测时需要的encoder
    encoder_model = Model(encoder_inputs, encoder_states)

    # 新序列预测时需要的decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    # 返回需要的三个模型
    return model, encoder_model, decoder_model

def predict_sequence(infenc, infdec, source, n_steps, cardinality):
    # 输入序列编码得到编码状态向量
    state = infenc.predict(source)
    # 初始目标序列输入：通过开始字符计算目标序列第一个字符，这里是0
    target_seq = array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
    # 输出序列列表
    output = list()
    for t in range(n_steps):
        # predict next char
        yhat, h, c = infdec.predict([target_seq] + state)
        # 截取输出序列，取后三个
        output.append(yhat[0, 0, :])
        # 更新状态
        state = [h, c]
        # 更新目标序列(用于下一个词预测的输入)
        target_seq = yhat
    return array(output)

# one_hot解码
def one_hot_decode(encoded_seq):
    return [argmax(vector) for vector in encoded_seq]


# 参数设置
n_features = 50 + 1
n_steps_in = 6
n_steps_out = 3
# 定义模型
train, infenc, infdec = define_models(n_input=n_features, n_output=n_features, n_units=128)
train.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
# 生成训练数据
X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 10000)
print(X1.shape, X2.shape, y.shape)


# 训练模型
history = train.fit([X1, X2], y, epochs=20,validation_split=0.2)


'''

*************************************ATTENTION*************************************************************

'''
print('Attetion.................................................')

def define_models(n_input=51, n_output=51, n_units=128):  #
    # encoder:
    encoder_inputs = tf.keras.Input(shape=X1.shape[1:])
    print(X1.shape[1:])
    encoder = LSTM(n_units, return_state=True, return_sequences=True)
    encoder_ = LSTM(n_units, return_state=True)
    enc_outputs, enc_state_h_, enc_state_c_ = encoder(encoder_inputs)
    enc_outputs_, enc_state_h, enc_state_c = encoder(encoder_inputs)
    # decoder
    dec_states_inputs = [enc_state_h, enc_state_c]

    decoder_inputs = tf.keras.Input(shape=X2.shape[1:])
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    dec_outputs, dec_state_h, dec_state_c = decoder_lstm(decoder_inputs, initial_state=dec_states_inputs)
    attention_output = tf.keras.layers.Attention()([dec_outputs, enc_outputs])
    dense_output_ = Dense(n_output, activation='softmax', name="dense")
    dense_outputs = dense_output_(attention_output)

    model = Model([encoder_inputs, decoder_inputs], dense_outputs)

    # 新序列预测时需要的encoder
    encoder_model = Model(encoder_inputs, dec_states_inputs)

    # 新序列预测时需要的decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = dense_output_(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model


# one_hot解码
def one_hot_decode(encoded_seq):
    return [argmax(vector) for vector in encoded_seq]


# 参数设置
n_features = 50 + 1
n_steps_in = 6
n_steps_out = 3
# 定义模型
train, infenc, infdec = define_models(n_features, n_features, 128)
train.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
# 生成训练数据
X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 10000)
print(X1.shape, X2.shape, y.shape)

history_1 = train.fit([X1, X2], y, epochs=20,validation_split=0.2)
