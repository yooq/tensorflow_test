import tensorflow as tf
from tensorflow import  keras
from tensorflow.keras import layers,Sequential

x = tf .random.uniform([4,8,100]) #随机生成，4个句子，每个句子80个单词，每个词向量100维

xt0 = x[:,0,:]

print(xt0.shape)

cell = layers.SimpleRNNCell(units=64)

out,xt1 = cell(xt0,[tf.zeros([4,64])])  #对于rnn来讲  out 与xt1是一样的

print(out.shape)




'''
模块化
'''
units =64
rnn = keras.Sequential([
    layers.SimpleRNN(units=units,dropout=0.5,return_sequences=True,unroll=True)
    ,layers.SimpleRNN(units=units,dropout=0.5)
])

model = rnn(x)
print(model)
