import tensorflow as tf
from  tensorflow.keras import layers


'''
save()保存的模型结果，它既保持了模型的图结构，又保存了模型的参数
save_weights()保存的模型结果，它只保存了模型的参数，但并没有保存模型的图结构.

问：为什么模型要保存成h5格式的
答：h5占用的空间小
'''

model = tf.keras.Sequential([
layers.Dense(64, activation='relu'),
layers.Dense(10, activation='softmax')])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(type(model))
# model.save_weights('./weights/model')
# model.load_weights('./weights/model')
# model.save_weights('./model.h5')
# model.load_weights('./model.h5')
# model.save('all_model.h5')
# model = tf.keras.models.load_model('all_model.h5')

'''
保存网路结构

'''
# 序列化成json
import json
import pprint
json_str = model.to_json()
pprint.pprint(json.loads(json_str))
fresh_model = tf.keras.models.model_from_json(json_str)  #加载


# 保持为yaml格式  #需要提前安装pyyaml
# yaml_str = model.to_yaml()
# print(yaml_str)
# fresh_model = tf.keras.models.model_from_yaml(yaml_str)
