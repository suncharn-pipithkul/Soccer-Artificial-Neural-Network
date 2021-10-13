import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
import math

# Functions generating random inputs (2D coordinates) for each class
# We are simply assuming normally distributed samples for each class

def get_example():
    return [np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0), np.random.uniform(0.0, 1.0)]

num_inputs = 3
num_outputs = 2

num_train_exemplars = 10000
num_test_exemplars = 500

x_train = []
y_train = []

np.random.seed(0)

for i in range(num_train_exemplars):
    x_train.append(get_example())
    y_train.append([x_train[-1][0] + x_train[-1][1] + x_train[-1][2], x_train[-1][0] + x_train[-1][1] - x_train[-1][2]])

x_train = np.array(x_train)
y_train = np.array(y_train)

x_test = []
y_test = []

for i in range(num_test_exemplars):
    x_test.append(get_example())
    y_test.append([x_test[-1][0] + x_test[-1][1] + x_test[-1][2], x_test[-1][0] + x_test[-1][1] - x_test[-1][2]])

x_test = np.array(x_test)
y_test = np.array(y_test)

input_shape = (num_inputs,)
batch_size = 20
num_epochs = 20

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# build the neural network model
model = Sequential()
# model.add(Dense(20, activation='relu', name='hidden'))
model.add(Dense(num_outputs, activation='linear', name='output'))

model.compile(loss='mse', optimizer=keras.optimizers.Adam(), metrics=['mse'])

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=2, validation_data=(x_test, y_test))

model.summary()

# Check the classification performance of the trained network on the test data 
final_train_loss, final_train_accuracy = model.evaluate(x_train, y_train, verbose=0)
final_test_loss, final_test_accuracy = model.evaluate(x_test, y_test, verbose=0)
 
print('Final training loss (mean square error):', final_train_loss)
print('Final test loss (mean square error):', final_test_loss)

print(model.predict([[0.2, 0.3, 0.1]]))

