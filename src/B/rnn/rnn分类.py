import  os
import  tensorflow as tf
import  numpy as np
from    tensorflow import keras
from    tensorflow.keras import layers
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


class MyRNN(keras.Model):
    def __init__(self, units):
        super(MyRNN, self).__init__()
        #[b,64]
        self.state0 = [tf.zeros([batchsz,units])]

        self.embeding = layers.Embedding(
            totalwords,embeding_len,input_length=max_review_len
        )


        self.rnn_cell0 = layers.SimpleRNNCell(units,dropout=0.5)

        self.out = layers.Dense(1)

    def call(self,inputs,training=None):
        x = inputs

        x = self.embeding(x)

        state0 = self.state0


        for word in tf.unstack(x,axis=1):  #[b,100] 一个词一个词进
            out,state1 = self.rnn_cell0(word,state0,training)
            # out1,state1 = self.rnn_cell1(word,state0,training)

            state0 =state1
        x = self.out(out)

        return x

def trian_main():
    units = 64
    epochs = 4
    model = MyRNN(units)
    model.build(input_shape=(None,100))
    model.summary()
    # model.compile(optimizer=keras.optimizers.Adam(0.001),
    #               loss=tf.losses.BinaryCrossentropy(),
    #               metrics=['accuracy'])
    # model.fit(db_train, epochs=epochs)
    #
    # # model.evaluate(db_test)
    #

if __name__ == '__main__':
    trian_main()

