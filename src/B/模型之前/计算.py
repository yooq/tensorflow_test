import tensorflow as tf


a = tf.ones([2,2])
an = tf.norm(a)  #二范数
print(an)

an_ = tf.norm(a,axis=1,ord=1)  #ord 表示范数类型，1范数，而范数
print(an_)

tf.reduce_mean
tf.reduce_max
tf.reduce_max
tf.reduce_all

# tf.argmax(a,axis=) #返回最大值的索引
tf.argmin()
tf.equal()
tf.unique() #去重
