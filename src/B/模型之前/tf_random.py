import tensorflow as tf

# 正态分布
rand = tf.random.normal([2,2],mean=1,stddev=1)  #随机生成正太分布，mean均值,stddev方差
print(rand.numpy())


# 截断正太随机但如果值的大小大于平均值的2个标准差，则删除并重新选取
# 防止初始化的时候就导致梯度消失，正太分布图像与sigmoid部分很像
rand2 = tf.random.truncated_normal([2,3],mean=1,stddev=1)


# 均匀分布
rand3 = tf.random.uniform([2,3],minval=0,maxval=10,dtype=tf.int32)
print(rand3)

print(tf.random.shuffle(rand3)) # 随机shuffle

y=tf.range(4)
print(tf.one_hot(y,depth=10)) #长度为10 的one-hot

loss = tf.keras.losses.mse(y,out)
loss = tf.reduce_mean(loss)


tf.embedding
