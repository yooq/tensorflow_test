'''

dataset 的元素是张量，可以是单个张量，张量元组

可以处理各种数据集合类型的数据
'''


from tensorflow import data
import tensorflow as tf

data = [1,2,3,4,5,6] # <TensorSliceDataset shapes: (), types: tf.int32>
data_arr = [[1,2],[3,4],[5,6],[7,8],[9,10]] # <TensorSliceDataset shapes: (2,), types: tf.int32>
data_dcit = {'A':[1],'B':[2],'C':[3]}  # <TensorSliceDataset shapes: {A: (), B: (), C: ()}, types: {A: tf.int32, B: tf.int32, C: tf.int32}>

data_set = tf.data.Dataset.from_tensor_slices(data)



'''
将tensor类型转换成numpy类型
'''
#
# for ele in data_set:
#     print(ele.numpy())


data_set = data_set.shuffle(buffer_size=2)  # 乱序

data_set = data_set.repeat(count=3)  # 且重复干 3 次这种乱序的事。。。目的，防止模型记住了数据的顺序
data_set = data_set.batch(batch_size=4)  # 批次
# # data_set.map()  可映射处理数据
for ele in data_set:
    print(ele.numpy())
