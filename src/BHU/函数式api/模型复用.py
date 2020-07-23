from tensorflow.keras.applications import VGG16
from tensorflow import keras
import numpy as np

vgg16=VGG16()

feature_list = [layer.output for layer in vgg16.layers]
feat_ext_model = keras.Model(inputs=vgg16.input, outputs=feature_list[-1])



img = np.random.random((1, 224, 224, 3)).astype('float32')
ext_features = feat_ext_model(img)
print(type(ext_features))

