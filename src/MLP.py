import tensorflow as tf
from  sklearn.datasets import load_boston

data = load_boston()

import pandas as pd
df = pd.DataFrame(data.data)

df_ = pd.DataFrame(data.target)
df__ =pd.concat([df,df_],axis=1)
print(df__)



model =tf.keras.Sequential()
model.add(tf.keras.layers.Dense(10,input_shape=(13,),activation=tf.keras.activations.relu))
model.add(tf.keras.layers.Dense(5,activation=tf.keras.activations.relu))
model.add(tf.keras.layers.Dense(1))

'''

梯度衰减
https://blog.csdn.net/Light2077/article/details/106629697?utm_medium=distribute.pc_relevant.none-task-blog-baidujs-1

  return initial_learning_rate * decay_rate ^ (step / decay_steps)

'''
# 学习率衰减
exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(
                        initial_learning_rate=0.1, decay_steps=1000, decay_rate=0.96,staircase=True)


model.compile(optimizer=tf.keras.optimizers.Adam(exponential_decay),loss=tf.keras.losses.mse)
history = model.fit(data.data,data.target,epochs=300)


pre = model.predict(df__.iloc[0:10,0:-1])

print(pre)

print(df__.iloc[0:10,-1])
