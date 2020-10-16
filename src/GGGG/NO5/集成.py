import numpy as np
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape([x_train.shape[0], -1])
x_test = x_test.reshape([x_test.shape[0], -1])
print(x_train.shape, ' ', y_train.shape)
print(x_test.shape, ' ', y_test.shape)

def mlp_model():
    model = keras.Sequential(
        [
            layers.Dense(64,activation='relu',input_shape=(784,)),
            layers.Dropout(0.2),
            layers.Dense(64,activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64,activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(10, activation='softmax')
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model1 = KerasClassifier(build_fn=mlp_model,epochs=100,verbose=0)
model1._estimator_type = 'classifier'
model2 = KerasClassifier(build_fn=mlp_model, epochs=100, verbose=0)
model2._estimator_type = 'classifier'
model3 = KerasClassifier(build_fn=mlp_model, epochs=100, verbose=0)
model3._estimator_type = 'classifier'
ensemble_clf = VotingClassifier(estimators=[
    ('model1', model1), ('model2', model2), ('model3', model3)
], voting='soft')
ensemble_clf.fit(x_train, y_train)

y_pred = ensemble_clf.predict(x_test)
print('acc: ', accuracy_score(y_pred, y_test))
