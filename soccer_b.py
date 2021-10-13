import keras
import pickle
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras import backend as K
from keras.models import load_model
import matplotlib.pyplot as plt
import random as rnd
import numpy as np

with open('soccer_data.pickle', 'rb') as f:
    (x_train, y_train), (x_test, y_test) = pickle.load(f)

# Scaling outputs to be larger than 0 and less than 1
# (Not exactly 0 or 1, because we want to use the sigmoid function that can only approach but never reach these values)
y_train_norm = (y_train - [-1, -1, 0, -1000000]) / [12, 12, 200, 3000000]
y_test_norm = (y_test - [-1, -1, 0, -1000000]) / [12, 12, 200, 3000000]

# build the neural network model
batch_size = 100
num_outputs = 4
epochs = 200
num_inputs = 22
input_shape = (num_inputs,)

model = Sequential()
model.add(Dense(100, input_shape=input_shape, activation='relu', name='hidden'))
model.add(Dense(num_outputs, activation='sigmoid', name='output'))

model.compile(loss=keras.losses.mse,
              optimizer=keras.optimizers.Adam(),
              metrics=['mse'])

model.summary()

# train and evaluate the model
model.fit(x_train, y_train_norm,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test_norm))

y_train_norm = (y_train + [-1, -1, 0, 1000000]) / [12, 12, 300, 3000000]
y_test_norm = (y_test + [-1, -1, 0, 1000000]) / [12, 12, 300, 3000000]

# Get the rescaled predictions for the test set and compute their deviation from the desired ones 
y_predict_norm = model.predict(x_test)

y_predict = y_predict_norm * [12, 12, 300, 3000000] - [1, 1, 0, 1000000]
y_diff = y_predict - y_test
y_stdev = np.std(y_diff, axis=0)

print(y_stdev)
