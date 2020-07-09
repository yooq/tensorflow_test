import  os
import  tensorflow as tf
import  numpy as np
from    tensorflow import keras
from  tensorflow.keras import layers,Sequential
from tensorflow.python.keras.datasets import imdb
from src.B.rnn import imb_data

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')




totalwords =10000
max_review_len =80
embeding_len = 100
batchsz = 128

(train_data, train_labels), (test_data, test_labels)=imb_data.imdb_data(num_words=totalwords)
print(train_data.shape)

print(set(train_labels),set(test_labels))


train_data = keras.preprocessing.sequence.pad_sequences(train_data,maxlen=max_review_len)  #取句子最大长度作为参考标准

test_data = keras.preprocessing.sequence.pad_sequences(test_data,maxlen=max_review_len)

db_train = tf.data.Dataset.from_tensor_slices((train_data,train_labels))
db_train = db_train.shuffle(10000).batch(batch_size=batchsz)
db_test = tf.data.Dataset.from_tensor_slices((test_data,test_labels))
db_test = db_test.batch(batchsz,drop_remainder=True)  #第二个参数防止生成较小的批


model = Sequential()
model.add(layers.Embedding(totalwords,32))
# model.add(layers.SimpleRNN(32))

# model.add(keras.layers.Bidirectional(layers.SimpleRNN(32), merge_mode='concat', weights=None))   Bidirectional表示双向

# 多层rnn
# 建议需要使用函数式变成，见Birnn
model.add(layers.Bidirectional(layers.LSTM(32, return_sequences=True)))
model.add(layers.Bidirectional(layers.LSTM(32)))


'''
SimpleRnn 可以不需要在中间序列指定输入 , return_sequences=True  返回中间结果
SimpleRnnCell 需要在中间序列指定输入
cell 可以做单独单元   rnn = tf.keras.layers.RNN(tf.keras.layers.LSTMCell(4))
'''
# model.add(layers.LSTM(32))
model.add(layers.Dense(1,activation='sigmoid'))

model.summary()

model.compile(optimizer='rmsprop',
             loss='binary_crossentropy',
             metrics=['acc'])

history = model.fit(train_data,train_labels,
                   epochs=10,
                   batch_size=128,
                   validation_split=0.2)

