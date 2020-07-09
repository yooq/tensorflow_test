import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import activations
from tensorflow.keras.layers import Layer, Input, Embedding, LSTM, Dense, Attention
from tensorflow.keras.models import Model

class Encoder(keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_units):
        super(Encoder, self).__init__()
        # Embedding Layer
        self.embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)
        # Encode LSTM Layer
        self.encoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True, name="encoder")

    def call(self, inputs):
        encoder_embed = self.embedding(inputs)
        encoder_outputs, state_h, state_c = self.encoder_lstm(encoder_embed)
        return encoder_outputs, state_h, state_c


class Decoder(keras.Model):
    def __init__(self, vocab_size, embedding_dim, hidden_units):
        super(Decoder, self).__init__()
        # Embedding Layer
        self.embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)
        # Decode LSTM Layer
        self.decoder_lstm = LSTM(hidden_units, return_sequences=True, return_state=True, name="decoder")
        # Attention Layer
        self.attention = Attention()

    def call(self, enc_outputs, dec_inputs, states_inputs):
        decoder_embed = self.embedding(dec_inputs)
        dec_outputs, dec_state_h, dec_state_c = self.decoder_lstm(decoder_embed, initial_state=states_inputs)
        attention_output = self.attention([dec_outputs, enc_outputs])

        return attention_output, dec_state_h, dec_state_c


def Seq2Seq(maxlen, embedding_dim, hidden_units, vocab_size):
    """
    seq2seq model
    """
    # Input Layer
    encoder_inputs = tf.keras.Input(shape=(maxlen,), name="encoder_input")
    decoder_inputs = tf.keras.Input(shape=(None,), name="decoder_input")

    # Encoder Layer
    encoder = Encoder(vocab_size, embedding_dim, hidden_units)
    enc_outputs, enc_state_h, enc_state_c = encoder(encoder_inputs)
    dec_states_inputs = [enc_state_h, enc_state_c]

    # Decoder Layer
    decoder = Decoder(vocab_size, embedding_dim, hidden_units)
    attention_output, dec_state_h, dec_state_c = decoder(enc_outputs, decoder_inputs, dec_states_inputs)

    # Dense Layer
    dense_outputs = Dense(vocab_size, activation='softmax', name="dense")(attention_output)

    # seq2seq model
    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=dense_outputs)
    return model


maxlen = 10
embedding_dim = 50
hidden_units = 128
vocab_size = 10000

model = Seq2Seq(maxlen, embedding_dim, hidden_units, vocab_size)


model.summary()


'''
查看encoder模块,获取encoder中间层
'''

def encoder_infer(model):
    encoder_model = Model(inputs=model.get_layer('encoder').input,outputs=model.get_layer('encoder').output)
    return encoder_model

encoder_model = encoder_infer(model)
print(encoder_model.summary())

# ****************

'''
查看decoder模块结构，获取decoder中间层
'''
def decoder_infer(model, encoder_model):
    encoder_output = encoder_model.get_layer('encoder').output[0]
    maxlen, hidden_units = encoder_output.shape[1:]

    dec_input = model.get_layer('decoder_input').input
    enc_output = Input(shape=(maxlen, hidden_units), name='enc_output')

    dec_input_state_h = Input(shape=(hidden_units,), name='input_state_h')
    dec_input_state_c = Input(shape=(hidden_units,), name='input_state_c')
    dec_input_states = [dec_input_state_h, dec_input_state_c]

    decoder = model.get_layer('decoder')
    dec_outputs, out_state_h, out_state_c = decoder(enc_output, dec_input, dec_input_states)
    dec_output_states = [out_state_h, out_state_c]

    decoder_dense = model.get_layer('dense')
    dense_output = decoder_dense(dec_outputs)

    decoder_model = Model(inputs=[enc_output, dec_input, dec_input_states],
                          outputs=[dense_output] + dec_output_states)
    return decoder_model

decoder_model = decoder_infer(model, encoder_model)
print(decoder_model.summary())


# 读取数据以及词典的方法
def read_vocab(vocab_path):
    vocab_words = []
    with open(vocab_path, "r", encoding="utf8") as f:
        for line in f:
            vocab_words.append(line.strip())
    return vocab_words

def read_data(data_path):
    datas = []
    with open(data_path, "r", encoding="utf8") as f:
        for line in f:
            words = line.strip().split()
            datas.append(words)
    return datas

def process_data_index(datas, vocab2id):
    data_indexs = []
    for words in datas:
        line_index = [vocab2id[w] if w in vocab2id else vocab2id["<UNK>"] for w in words]
        data_indexs.append(line_index)
    return data_indexs


# 预处理数据并生成词典
vocab_words = read_vocab("./data/ch_word_vocab.txt")
special_words = ["<PAD>", "<UNK>", "<GO>", "<EOS>"]
vocab_words = special_words + vocab_words
vocab2id = {word: i for i, word in enumerate(vocab_words)}
id2vocab = {i: word for i, word in enumerate(vocab_words)}

num_sample = 10000
source_data = read_data("./data/ch_source_data_seg.txt")[:num_sample]
source_data_ids = process_data_index(source_data, vocab2id)
target_data = read_data("./data/ch_target_data_seg.txt")[:num_sample]
target_data_ids = process_data_index(target_data, vocab2id)

print("vocab test: ", [id2vocab[i] for i in range(10)])
print("source test: ", source_data[10])
print("source index: ", source_data_ids[10])
print("target test: ", target_data[10])
print("target index: ", target_data_ids[10])


#Decoder部分输入输出加上开始结束标识
def process_decoder_input_output(target_indexs, vocab2id):
    decoder_inputs, decoder_outputs = [], []
    for target in target_indexs:
        decoder_inputs.append([vocab2id["<GO>"]] + target)
        decoder_outputs.append(target + [vocab2id["<EOS>"]])
    return decoder_inputs, decoder_outputs

target_input_ids, target_output_ids = process_decoder_input_output(target_data_ids, vocab2id)
print("decoder inputs: ", target_input_ids[:2])
print("decoder outputs: ", target_output_ids[:2])


# 数据pad填充
maxlen = 10
source_input_ids = keras.preprocessing.sequence.pad_sequences(source_data_ids, padding='post', maxlen=maxlen)
target_input_ids = keras.preprocessing.sequence.pad_sequences(target_input_ids, padding='post',  maxlen=maxlen)
target_output_ids = keras.preprocessing.sequence.pad_sequences(target_output_ids, padding='post',  maxlen=maxlen)
print(source_data_ids[:5])
print(target_input_ids[:5])
print(target_output_ids[:5])


# 构建模型
K.clear_session()

maxlen = 10
embedding_dim = 50
hidden_units = 128
vocab_size = len(vocab2id)

model = Seq2Seq(maxlen, embedding_dim, hidden_units, vocab_size)


#训练模型
epochs = 10
batch_size = 32
val_rate = 0.2

loss_fn = keras.losses.SparseCategoricalCrossentropy()
model.compile(loss=loss_fn, optimizer='adam')
# fit 参数下 inputs=[encoder_inputs, decoder_inputs], outputs=dense_outputs
model.fit([source_input_ids, target_input_ids], target_output_ids,
          batch_size=batch_size, epochs=epochs, validation_split=val_rate)


#模型预测

import numpy as np
maxlen = 10

def infer_predict(input_text, encoder_model, decoder_model):
    text_words = input_text.split()[:maxlen]
    input_id = [vocab2id[w] if w in vocab2id else vocab2id["<UNK>"] for w in text_words]
    # input_id = [vocab2id["<START>"]] + input_id + [vocab2id["<END>"]]
    if len(input_id) < maxlen:
        input_id = input_id + [vocab2id["<PAD>"]] * (maxlen - len(input_id))

    input_source = np.array([input_id])
    input_target = np.array([vocab2id["<GO>"]])

    # 编码器encoder预测输出
    enc_outputs, enc_state_h, enc_state_c = encoder_model.predict([input_source])

    dec_inputs = input_target
    dec_states_inputs = [enc_state_h, enc_state_c]

    result_id = []
    result_text = []

    for i in range(maxlen):
        # 解码器decoder预测输出
        dense_outputs, dec_state_h, dec_state_c = decoder_model.predict([enc_outputs, dec_inputs] + dec_states_inputs)
        print(dense_outputs.shape)
        print(dense_outputs)
        pred_id = np.argmax(dense_outputs[0][0])
        result_id.append(pred_id)
        result_text.append(id2vocab[pred_id])
        if id2vocab[pred_id] == "<EOS>":
            break
        dec_inputs = np.array([[pred_id]])
        dec_states_inputs = [dec_state_h, dec_state_c]
    return result_id, result_text


input_sent = "你 在 干 什么 呢"
result_id, result_text = infer_predict(input_sent, encoder_model, decoder_model)

print("Input: ", input_sent)
print("Output: ", result_text, result_id)
















# 预测好的方法
print('beam search....')
import heapq

def infer_encoder_output(input_text, encoder, maxlen=7):
    text_words = input_text.split()[:maxlen]
    input_id = [vocab2id[w] if w in vocab2id else vocab2id["<UNK>"] for w in text_words]
    input_id = [vocab2id["<GO>"]] + input_id + [vocab2id["<EOS>"]]
    if len(input_id) < maxlen:
        input_id = input_id + [vocab2id["<PAD>"]] * (maxlen - len(input_id))
    input_source = np.array([input_id])
    # 编码器encoder预测输出
    enc_outputs, enc_state_h, enc_state_c = encoder.predict([input_source])
    enc_state_outputs = [enc_state_h, enc_state_c]
    return enc_outputs, enc_state_outputs

def infer_beam_search(enc_outputs, enc_state_outputs, decoder, k=5):
    dec_inputs = [vocab2id["<GO>"]]
    states_curr = {0: enc_state_outputs}
    seq_scores = [[dec_inputs, 1.0, 0]]

    for _ in range(maxlen):
        cands = list()
        states_prev = states_curr
        for i in range(len(seq_scores)):
            seq, score, state_id = seq_scores[i]
            dec_inputs = np.array(seq[-1:])
            dec_states_inputs = states_prev[state_id]
            # 解码器decoder预测输出
            dense_outputs, dec_state_h, dec_state_c = decoder.predict([enc_outputs, dec_inputs] + dec_states_inputs)
            prob = dense_outputs[0][0]
            states_curr[i] = [dec_state_h, dec_state_c]

            for j in range(len(prob)):
                cand = [seq + [j], score * prob[j], i]
                cands.append(cand)

        seq_scores = heapq.nlargest(k, cands, lambda d: d[1])

    res = " ".join([id2vocab[i] for i in seq_scores[0][0]])
    return res

enc_outputs, enc_state_outputs = infer_encoder_output(input_sent,encoder_model,maxlen=7)

res = infer_beam_search(enc_outputs,enc_state_outputs,decoder_model,k=5)
print(res)
