import tensorflow as tf

a = tf.random.uniform([2,3],maxval=10,minval=1,dtype=tf.int32)

aa = tf.pad(a,[[1,3],[1,2]])   #[[a,b],[m,n]]  a,b表示行，a表示上面行，b表示下面行。。。m,n表示列，m表示左边列，n表示右边列
print(aa)

# 针对多为的tensor.   #[[a,b],[m,n]]   可以括号成  [[a,b],[m,n],[i,j],[l,k]]

# tensor复制扩充

cc= tf.tile(a,[2,2])
print(cc.shape)

# [[5 8 7 5 8 7]
#  [1 4 9 1 4 9]
#  [5 8 7 5 8 7]
#  [1 4 9 1 4 9]]

ccc = tf.broadcast_to(a,[2,2,3])
print(ccc)
