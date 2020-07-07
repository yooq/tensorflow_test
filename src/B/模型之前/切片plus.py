import  tensorflow as tf


# gather   和    gather_nd
a = tf.random.uniform([4,35,8],minval=1,maxval=50,dtype=tf.int32)
print(a)
a_0  = tf.gather(a,axis=0,indices=[2,3]) #在第一维上 拿出索引为2和3的数据,#收集 ,按照tf_2的索引，来收集tf_中的数据
print(a_0.shape) #（2,35,8）

a_1 = tf.gather(a,axis=1,indices=[1,5,2,7,9])
print(a_1.shape) #(4, 5, 8)

a_2 = tf.gather(a,axis=2,indices=[3,6,5])
print(a_2.shape) #(4, 35, 3)

a_ = tf.gather_nd(a,[
                        [[0,0,0],
                        [1,1,1],
                        [2,2,2]]
])  #取第 i 维第 i 行第 i 列的数据

print(a_)



# boolean_mask
a_m = tf.boolean_mask(a,mask=[True,True,False,False],axis=0) #true 返回false不返回
print(a_m.shape)

# a_m_1= tf.boolean_mask(a,mask=[[True,True,False,False],[第二维]])

# 转置

tf.transpose(a,perm=[0,1,2]) #各维度相互转换


# 增加维度，减少维度
print(tf.expand_dims(a,axis=0).shape) #(4, 35, 8) ---->(1, 4, 35, 8)
print(tf.expand_dims(a,axis=1).shape) #(4, 35, 8) ---->(4, 1, 35, 8)

# 减少维度只能减少shape=1的维度
aaa = tf.random.uniform([1,2,2],maxval=10,minval=1,dtype=tf.int32)
print(tf.squeeze(aaa,axis=0).shape) #(1,2, 2)---->(2, 2)

# with tf.GradientTape
