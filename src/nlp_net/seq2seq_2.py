'''
步骤：定义完整的训练的模型------>拆开解码和编码部分用于预测------------->训练及预测

'''

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,LSTM,Dense
import numpy as np
import tensorflow as tf




batch_size = 64
epochs = 10
latent_dim = 256
num_samples = 10000
data_path = 'ell.txt'

# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')

for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text, _ = line.split('\t')
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)


input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])

target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

print('input_texts_len:', len(input_texts))
print('num_encoder_tokens:', num_encoder_tokens)
print('num_decoder_tokens:', num_decoder_tokens)
print('max_encoder_seq_length:', max_encoder_seq_length)
print('max_decoder_seq_length:', max_decoder_seq_length)
print(input_token_index)
print(target_token_index)


# encoder
encoder_input_data = np.zeros(
    (len(input_texts),max_encoder_seq_length,num_encoder_tokens),dtype='float32'
) #多少条句子---最长的句子多少词---单个词编码长度

# decoder
decoder_input_data = np.zeros(
    (len(input_texts),max_decoder_seq_length,num_decoder_tokens),dtype='float32'
)

decoder_target_data = np.zeros(
    (len(input_texts),max_decoder_seq_length,num_decoder_tokens)
)


for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    encoder_input_data[i, t + 1:, input_token_index[' ']] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.
    decoder_input_data[i, t + 1:, target_token_index[' ']] = 1.
    decoder_target_data[i, t:, target_token_index[' ']] = 1.

# 模型

class Encoder(tf.keras.Model):
    def __init__(self,latent_dim):
        super(Encoder,self).__init__()
        self.encoder_lstm = LSTM(latent_dim,return_state=True)

    def call(self,inputs,training=True):
        encoder_outputs, state_h, state_c = self.encoder_lstm(inputs)
        return encoder_outputs,[state_h,state_c]

class Decoder(tf.keras.Model):
    def __init__(self,latent_dim):
        super(Decoder, self).__init__()
        self.decoder_lstm = LSTM(latent_dim,return_sequences=True,return_state=True)
    def call(self,inputs,initial_state):
        decoder_outputs, _1, _2 = self.decoder_lstm(inputs, initial_state=initial_state)
        return decoder_outputs,[_1,_2]


def seq2seq(latent_dim,num_decoder_tokens):
    encoder_inputs = Input(shape=(None, num_encoder_tokens), name='encoder_inputs')
    decoder_inputs = Input(shape=(None, num_decoder_tokens), name='decoder_inputs')

    encoder = Encoder(latent_dim)
    _,state_ = encoder(encoder_inputs)

    decoder = Decoder(latent_dim)
    decoder_outputs,_ = decoder(decoder_inputs,state_)

    decoder_outputs = Dense(num_decoder_tokens, activation='softmax', name='dense')(decoder_outputs)
    model = Model(inputs=[encoder_inputs,decoder_inputs],outputs=decoder_outputs)
    return model

model = seq2seq(latent_dim,num_decoder_tokens)
model.summary()


print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
# 用于预测
def encoder(model):
   encoder = Model(inputs = model.get_layer('encoder').input,outputs = model.get_layer('encoder').output)
   return encoder

encoder_model = encoder(model)
encoder_model.summary()


def decoder(model,encoder_model):

    dec_input_state_h = Input(shape=(latent_dim,), name='input_h')
    dec_input_state_c = Input(shape=(latent_dim,), name='input_c')
    dec_states_input = [dec_input_state_h, dec_input_state_c]


    decoder_input = Input(shape=(None, num_decoder_tokens))

    decoder = model.get_layer('decoder')
    decoder_outputs,state = decoder(decoder_input,dec_states_input)
    dense = model.get_layer('dense')
    dense_decoder = dense(decoder_outputs)
    dec_model = Model(inputs=[decoder_input,dec_states_input],outputs=[dense_decoder]+state)

    return dec_model

print('_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+_+')
decoder_model =decoder(model,encoder_model)
decoder_model.summary()







epochs = 10
val_rate = 0.2

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
# fit 参数下 inputs=[encoder_inputs, decoder_inputs], outputs=dense_outputs


import os
if not os.path.exists('s2s5.h5'):
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2)
    model.save_weights('s2s5.h5')


model = model.load_weights('s2s5.h5')
model = seq2seq(latent_dim,num_decoder_tokens)
model.summary()







reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # 先把上句输入编码器得到编码的中间向量，这个中间向量将是解码器的初始状态向量
    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1,1,num_decoder_tokens))
    # 起始值
    target_seq[0,0,target_token_index['\t']]=1.
    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        # 把当前的解码器输入和当前的解码器状态向量送进解码器
        # 得到对下一个时刻的预测和新的解码器状态向量
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # 采样出概率最大的那个字作为下一个时刻的输入
        sampled_token_index = np.argmax(output_tokens[0, -1, :]) #最可能的字符的下标
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # 如果采样到了结束符或者生成的句子长度超过了decoder_len，就停止生成
        if (sampled_char == '\n' or
                len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # 否则我们更新下一个时刻的解码器输入和解码器状态向量
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


# 预测
for seq_index in range(100):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)



