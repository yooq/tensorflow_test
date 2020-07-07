import tensorflow as tf

x_train,x_test = tf.split(x,num_or_size_splits=[5000,1000])
y_train,y_test = tf.split(y,num_or_size_splits=[5000,1000])


model =tf.keras.Sequential()


    # ...  模型结构等


#  validation_split=0.1  表示十折交叉验证，0.9,0.1
# 这时候不需要validation_data，因为它就是那个0.1
model.fit(....,validation_data=(x_test,y_test),validation_freq=4
          # ,validation_split=0.1
          )


