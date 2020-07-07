import tensorflow as tf
from tensorflow.keras import layers


# 4 有多少卷积核表示output数量
layer  =  layers.Conv2D(4,kernel_size=5,strides=1,padding='vaild')

layer(x)
# 查看卷积参数
layer.kernel
layer.bias
