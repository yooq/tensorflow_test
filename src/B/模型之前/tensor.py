import tensorflow as tf
import numpy as np

a= np.arange(5)
aa = tf.convert_to_tensor(a)  #numpy等 转为tensor
print(aa) #tf.Tensor([0 1 2 3 4], shape=(5,), dtype=int64)

aa_float = tf.cast(aa,dtype=tf.float32) #修改tensor数据类型
print(aa_float) #tf.Tensor([0. 1. 2. 3. 4.], shape=(5,), dtype=float32)

print(tf.cast([0,1],dtype=tf.bool)) #tf.Tensor([False  True], shape=(2,), dtype=bool)

print(tf.is_tensor(a)) #False

aa.numpy()  #将tensor转换成numpy

print(tf.zeros_like(aa))

print('****************')



tf_ = tf.random.shuffle(a)
print(tf_) # tf.Tensor([0 3 1 2 4], shape=(5,), dtype=int64)
tf_1 = tf.sort(tf_,direction='DESCENDING')
print(tf_1) # tf.Tensor([4 3 2 1 0], shape=(5,), dtype=int64)
tf_2 = tf.argsort(tf_,direction='DESCENDING')  #排序后的原索引
print(tf_2) # tf.Tensor([4 1 3 2 0], shape=(5,), dtype=int32)

tf.gather(tf_,tf_2)  #收集 ,按照tf_2的索引，来收集tf_中的数据

res = tf.math.top_k(a,k=2)
# res.indices,---->类似 argsort
# res.values ,----->类似 sort
