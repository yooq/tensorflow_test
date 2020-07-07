import tensorflow as tf

#tf.where  配合grather_nd抽取元素


a = tf.random.normal([3,3])

mask=a>0
aa =tf.boolean_mask(a,mask=mask)
print(aa)

bb = tf.where(mask)
print(bb)


cc= tf.gather_nd(a,bb)
print(cc)  #等于 aa =tf.boolean_mask(a,mask=mask)

# tf.where (mask,A,B)
# mask是一个bool 矩阵。ture时从A矩阵取对应数据，false时从B矩阵取对应数据

# tf.scatter_nd(indices,updates,shape)

# 根据indices下标，将update中数据，更新到大小为shape矩阵中去
indices = tf.constant([[4],[3],[1],[7]])
updates = tf.constant([9,10,11,12])
s = tf .constant([15])
print(s)
ss = tf.scatter_nd(indices,updates,shape=s)
print(ss)


# m = tf.zeros([4,4,4])
# indices=tf.constant([0],[2])  #在第一维度中，索引为0，和2 的数据上更新，跟新的数据必须是shpe=[4,4]



# meshgrid  画图前的网格

x =tf.linspace(-2.,2,5)
y = tf.linspace(-2.,2,5)
print(x)
point_x,point_y =tf.meshgrid(x,y)
print(point_x.shape) #(5, 5)

p = tf.stack([point_x,point_y],axis=2)
print(p)

