import tensorflow as tf

# 常规索引,等价
one = tf.random.uniform([5,5,3],maxval=100,minval=1,dtype=tf.int32)
print(one)
print(one[0][0])
print(one[0,0])
print(one[-1:])
one[0,:,:,:]
# 等价
one[0,...]
# 还可以有
one[0,1,...]
one[0,...,1,0]
