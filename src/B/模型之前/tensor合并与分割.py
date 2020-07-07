import  tensorflow as tf
a = tf.ones([2,35,8])
b = tf.ones([3,35,8])


# 合并
c = tf .concat([a,b],axis=0)
print(c.shape)

# ****************************
a = tf.ones([3,3,8])
b = tf.ones([3,35,8])
c = tf .concat([a,b],axis=1)
print(c.shape)

# ****************************
a = tf.ones([4,35,8])
b = tf.ones([4,35,8])
c = tf.stack([a,b],axis=0) #维度一样的tensor,新增一维后合并
print(c.shape)

# unstack  是拆开

cc= tf.unstack(c,axis=3) #对应得那一维的值是多少，就拆多少个tensor,如这个对应得是8==(2, 4, 35, 8)，那就拆成8个tensor==(2, 4, 35)
print(len(cc))


# split指定长度拆开

tf.split(c,axis=3,num_or_size_splits=[2,2,4]) #在axis=3上，将维度8切成2,2,4三个值，得到三个tensor
