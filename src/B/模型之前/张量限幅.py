import tensorflow as tf

a1 = tf.random.uniform([10,],maxval=20,minval=-10,dtype=tf.int32)
print(a1)

# 使得最小值不小于阈值1，最大值不大于阈值2，用阈值代替

a = tf.maximum(a1,8) # 不能小于8
# print(a)
a = tf.minimum(a,10)  #不能超过10
# print(a)

# 取值范围2~8
a_ = tf.clip_by_value(a1,2,8)
print(a_)


# 故relu可表示为

a_r = tf.nn.relu(a1)
print(a_r)

# 等价于

a_m = tf.maximum(a1,0)
print(a_m)


# 范数裁剪，模裁剪

an = tf.random.normal([2,2],mean=10)

tf.norm(an) # 求模

aan = tf.clip_by_norm(an,15)  #截取模长度为15，不改变方向
print(aan)
print(tf.norm(aan)) #模为15


new,totalnorm = tf.clip_by_global_norm(an,5) #保持方向不变，且切割比例一样
print(new)
