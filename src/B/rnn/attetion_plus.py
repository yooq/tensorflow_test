import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

class Encoder(keras.Model):
    def __init__(self,vocab_size,embedding_dim,hidden_units):
        super(Encoder,self).__init__()

        # embedding
        self.enc_embedding = keras.layers.Embedding(vocab_size,embedding_dim,mask_zero=True)

        # encoder
        self.encoder = keras.layers.LSTM(hidden_units,return_sequences=True,return_state=True) #控制hidden,cell的两个参数

    def call(self,inputs,training=None):
        embedding = self.enc_embedding(inputs)
        enc_outputs,enc_h,enc_c = self.encoder(embedding)
        return enc_outputs,enc_h,enc_c


class Decoder(keras.Model):
    def __init__(self,vocab_size,embedding_dim,hidden_units):
        super(Decoder,self).__init__()
        self.dec_embedding = layers.Embedding(vocab_size,embedding_dim,mask_zero=True)
        self.decoder = layers.LSTM(hidden_units,return_state=True,return_sequences=True)
        self.attention = layers.Attention()

    def call(self,enc_output,dec_inputs,state_inputs):
        dec_embedding = self.dec_embedding(dec_inputs)
        dec_output,dec_h,dec_c = self.decoder(dec_embedding)
        dec_attention = self.attention([dec_output,enc_output])
        return dec_attention,dec_h,dec_c


def seq2seq(maxlen,embedding_dim,hidden_units,vocab_size):
    enc_input = tf.keras.Input(shape=(maxlen,), name="encoder_input")
    dec_input = tf.keras.Input(shape=(None,), name="decoder_input")
    encoder = Encoder(vocab_size=vocab_size,embedding_dim=embedding_dim,hidden_units=hidden_units)
    enc_outputs,enc_h,enc_c = encoder(enc_input)
    dec_states_inputs = [enc_h,enc_c] #转接状态

    # Decoder Layer
    decoder = Decoder(vocab_size, embedding_dim, hidden_units)
    attention_output, dec_state_h, dec_state_c = decoder(enc_outputs, dec_input
                                                         , dec_states_inputs
                                                         )

    # 全连接层
    des_outputs = layers.Dense(vocab_size,activation='softmax',name='dense')(attention_output)
    model =keras.Model(inputs=[enc_input,dec_input],outputs=des_outputs)
    return model

model = seq2seq(maxlen=10,embedding_dim=50,hidden_units=128,vocab_size=10000)
model.summary()


# 用于预测
def encoder(model):
   encoder = keras.Model(inputs = model.get_layer('encoder').input,outputs = model.get_layer('encoder').output)
   return encoder

encoder_model = encoder(model)
# encoder_model.summary()

# 用于预测
def decoder(model,encoder):
    encoder_output = encoder.get_layer('encoder').output[0]
    maxlen,hidden_units = encoder_output.shape[1:]

    dec_input = model.get_layer('decoder_input').input
    enc_output = keras.Input(shape=(maxlen,hidden_units),name='enc_output')

    dec_input_state_h = keras.Input(shape=(hidden_units,),name='input_h')
    dec_input_state_c = keras.Input(shape=(hidden_units,), name='input_c')
    dec_states_input =[dec_input_state_h,dec_input_state_c]

    decoder = model.get_layer('decoder')

    dec_output,out_dec_h,out_dec_c = decoder(enc_output,dec_input
                                             ,dec_states_input
                                             )
    dec_output_states = [out_dec_h,out_dec_c]

    dec_dense = model.get_layer('dense')
    dense_output =dec_dense(dec_output)

    dec_model =keras.Model(inputs=[enc_output,dec_input,dec_states_input],outputs=[dense_output]+dec_output_states)

    return dec_model

decoder_model =decoder(model,encoder_model)

#
# # 读取数据以及词典的方法
# def read_vocab(vocab_path):
#     vocab_words = []
#     with open(vocab_path, "r", encoding="utf8") as f:
#         for line in f:
#             vocab_words.append(line.strip())
#     return vocab_words
#
# def read_data(data_path):
#     datas = []
#     with open(data_path, "r", encoding="utf8") as f:
#         for line in f:
#             words = line.strip().split()
#             datas.append(words)
#     return datas
#
# def process_data_index(datas, vocab2id):
#     data_indexs = []
#     for words in datas:
#         line_index = [vocab2id[w] if w in vocab2id else vocab2id["<UNK>"] for w in words]
#         data_indexs.append(line_index)
#     return data_indexs
#
#
# # 预处理数据并生成词典
# vocab_words = read_vocab("./data/ch_word_vocab.txt")
# special_words = ["<PAD>", "<UNK>", "<GO>", "<EOS>"]
# vocab_words = special_words + vocab_words
# vocab2id = {word: i for i, word in enumerate(vocab_words)}
# id2vocab = {i: word for i, word in enumerate(vocab_words)}
#
# num_sample = 10000
# source_data = read_data("./data/ch_source_data_seg.txt")[:num_sample]
# source_data_ids = process_data_index(source_data, vocab2id)
# target_data = read_data("./data/ch_target_data_seg.txt")[:num_sample]
# target_data_ids = process_data_index(target_data, vocab2id)
#
# print("vocab test: ", [id2vocab[i] for i in range(10)])
# print("source test: ", source_data[10])
# print("source index: ", source_data_ids[10])
# print("target test: ", target_data[10])
# print("target index: ", target_data_ids[10])
#
#
# #Decoder部分输入输出加上开始结束标识
# def process_decoder_input_output(target_indexs, vocab2id):
#     decoder_inputs, decoder_outputs = [], []
#     for target in target_indexs:
#         decoder_inputs.append([vocab2id["<GO>"]] + target)
#         decoder_outputs.append(target + [vocab2id["<EOS>"]])
#     return decoder_inputs, decoder_outputs
#
# target_input_ids, target_output_ids = process_decoder_input_output(target_data_ids, vocab2id)
# print("decoder inputs: ", target_input_ids[:2])
# print("decoder outputs: ", target_output_ids[:2])
#
#
# # 数据pad填充
# maxlen = 10
# source_input_ids = keras.preprocessing.sequence.pad_sequences(source_data_ids, padding='post', maxlen=maxlen)
# target_input_ids = keras.preprocessing.sequence.pad_sequences(target_input_ids, padding='post',  maxlen=maxlen)
# target_output_ids = keras.preprocessing.sequence.pad_sequences(target_output_ids, padding='post',  maxlen=maxlen)
# print(source_data_ids[:5])
# print(target_input_ids[:5])
# print(target_output_ids[:5])
#
# from tensorflow.keras import backend as K
# K.clear_session()
# maxlen = 10
# embedding_dim = 50
# hidden_units = 128
# vocab_size = len(vocab2id)
#
# model = seq2seq(maxlen, embedding_dim, hidden_units, vocab_size)
# epochs = 100
# batch_size = 32
# val_rate = 0.2
#
#
# loss_fn = keras.losses.SparseCategoricalCrossentropy()
# model.compile(loss=loss_fn, optimizer='adam')
# # fit 参数下 inputs=[encoder_inputs, decoder_inputs], outputs=dense_outputs
# model.fit([source_input_ids, target_input_ids], target_output_ids,
#           batch_size=batch_size, epochs=epochs, validation_split=val_rate)
#
#
# # 预测好的方法
# print('beam search....')
# import heapq
# import numpy as np
#
# def infer_encoder_output(input_text, encoder, maxlen=7):
#     text_words = input_text.split()[:maxlen]
#     input_id = [vocab2id[w] if w in vocab2id else vocab2id["<UNK>"] for w in text_words]
#     input_id = [vocab2id["<GO>"]] + input_id + [vocab2id["<EOS>"]]
#     if len(input_id) < maxlen:
#         input_id = input_id + [vocab2id["<PAD>"]] * (maxlen - len(input_id))
#     input_source = np.array([input_id])
#     # 编码器encoder预测输出
#     enc_outputs, enc_state_h, enc_state_c = encoder.predict([input_source])
#     enc_state_outputs = [enc_state_h, enc_state_c]
#     return enc_outputs, enc_state_outputs
#
# def infer_beam_search(enc_outputs, enc_state_outputs, decoder, k=5):
#     dec_inputs = [vocab2id["<GO>"]]
#     states_curr = {0: enc_state_outputs}
#     seq_scores = [[dec_inputs, 1.0, 0]]
#
#     for _ in range(maxlen):
#         cands = list()
#         states_prev = states_curr
#         for i in range(len(seq_scores)):
#             seq, score, state_id = seq_scores[i]
#             dec_inputs = np.array(seq[-1:])
#             dec_states_inputs = states_prev[state_id]
#             # 解码器decoder预测输出
#             dense_outputs, dec_state_h, dec_state_c = decoder.predict([enc_outputs, dec_inputs] + dec_states_inputs)
#             prob = dense_outputs[0][0]
#             states_curr[i] = [dec_state_h, dec_state_c]
#
#             for j in range(len(prob)):
#                 cand = [seq + [j], score * prob[j], i]
#                 cands.append(cand)
#
#         seq_scores = heapq.nlargest(k, cands, lambda d: d[1])
#
#     res = " ".join([id2vocab[i] for i in seq_scores[0][0]])
#     return res
#
# input_sent = "你 在 干 什么 呢"
# enc_outputs, enc_state_outputs = infer_encoder_output(input_sent,encoder_model,maxlen=7)
#
# res = infer_beam_search(enc_outputs,enc_state_outputs,decoder_model,k=5)
# print(input_sent)
# print(res)
