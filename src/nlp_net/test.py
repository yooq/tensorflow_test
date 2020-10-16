from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,LSTM,Dense
import numpy as np
import tensorflow as tf
model = tf.keras.models.load_model('s2s3.h5')
