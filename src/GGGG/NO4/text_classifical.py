import tensorflow as tf
from  tensorflow.keras import layers
from tensorflow import keras
import numpy as np
imdb=keras.datasets.imdb
(train_x, train_y), (test_x, text_y)=keras.datasets.imdb.load_data(num_words=10000)


# print("Training entries: {}, labels: {}".format(len(train_x), len(train_y)))
#
# print(train_x[0])
# print('len: ',len(train_x[0]), len(train_x[1]))

# {(词，词index)}
word_index = imdb.get_word_index()
word2id = {k:(v+3) for k,v in word_index.items()}
word2id['<PAD>'] = 0
word2id['<START>'] = 1
word2id['<UNK>'] = 2
word2id['<UNUSED>'] = 3

#{(词index,词)}
id2word = {v:k for k,v in word2id.items()}
def get_words(sent_ids):
    return ' '.join([id2word.get(i, '?') for i in sent_ids])

sent = get_words(train_x[0])
print('--------------')

train_x = keras.preprocessing.sequence.pad_sequences(
    train_x,value=word2id['<PAD>'],
    padding='post',maxlen=256
)#padding：‘pre’或‘post’，确定当需要补0时，在序列的起始还是结尾补
test_x = keras.preprocessing.sequence.pad_sequences(
    test_x,value=word2id['<PAD>'],
    padding='post',maxlen=256
)


from tensorflow.keras import layers
vocab_size = 10000
model = keras.Sequential()
model.add(layers.Embedding(vocab_size,16)) #None,None,16
model.add(layers.GlobalAveragePooling1D()) #均池化 bacth_size，词数，词向量长度。---->将词数和词向量做均池化 得到None,16
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(2,activation='softmax'))
model.summary()
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])


x_val = train_x[:10000]
x_train = train_x[10000:]

y_val = train_y[:10000]
y_train = train_y[10000:]

history = model.fit(x_train,y_train,
                   epochs=40, batch_size=512,
                   validation_data=(x_val, y_val),
                   verbose=1)

result = model.evaluate(test_x, text_y)
print(result)


# 单条数据预测
print(np.shape([test_x[1]])) #(1, 256)
print(np.shape(test_x)) #(25000, 256)
x_1 = np.reshape(test_x[1],newshape=(1,256,))
print(np.shape(x_1)) #(1, 256)
pre = model.predict(x_1)
print(len(pre))
print(pre)
print('====',np.argmax(pre[0]))


import matplotlib.pyplot as plt

history_dict = history.history
history_dict.keys()
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(acc)+1)

plt.plot(epochs, loss, 'bo', label='train loss')
plt.plot(epochs, val_loss, 'b', label='val loss')
plt.title('Train and val loss')
plt.xlabel('Epochs')
plt.xlabel('loss')
plt.legend()
plt.show()


plt.clf()   # clear figure

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

