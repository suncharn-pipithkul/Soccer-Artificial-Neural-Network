import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
# ===================================================================================================== #
# # Method 1: Regression style
# def desired_output(input_list):
#     if sum(input_list) < 6:
#         return sum(input_list)
#     else:
#         return 11 - sum(input_list)
#
# desired_output_scale = 100
#
# num_inputs = 11
# num_outputs = 1
#
# num_train_exemplars = 5000
# num_test_exemplars = 100
#
# # create training and test exemplars
# x_train = []
# y_train = []
# x_test = []
# y_test = []
#
# # creating training exemplars
# for i in range(num_train_exemplars):
#     chance = np.random.uniform(0.0, 1.0)
#     x_train.append(np.random.choice([0, 1], size=(11,), p=[chance, 1 - chance]))
#     y_train.append(desired_output(x_train[i]) / desired_output_scale)
# x_train = np.array(x_train)
# y_train = np.array(y_train)
# print('training exemplar created')
#
# # creating testing exemplars
# for i in range(num_test_exemplars):
#     chance = np.random.uniform(0.0, 1.0)
#     x_test.append(np.random.choice([0, 1], size=(11,), p=[chance, 1 - chance]))
#     y_test.append(desired_output(x_test[i]) / desired_output_scale)
# x_test = np.array(x_test)
# y_test = np.array(y_test)
# print('testing exemplar created')
#
# input_shape = (num_inputs,)
# batch_size = 10
# num_epochs = 100
#
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')
#
# # build the neural network model
# model = Sequential()
# model.add(Dense(200, activation='relu', name='hidden'))
# model.add(Dense(num_outputs, activation='linear', name='output'))
# model.compile(loss='mse', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
#
# # fit the model to the dataset
# history = model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=2, validation_data=(x_test, y_test))
#
# # evaluate the model
# final_train_loss, final_train_accuracy = model.evaluate(x_train, y_train, verbose=0)
# final_test_loss, final_test_accuracy = model.evaluate(x_test, y_test, verbose=0)
#
# print("predicted")
# print(model.predict([[0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0]]))
# print(model.predict([[1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0]]))
# print(model.predict([[1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1]]))
# print(model.predict([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
#
# print("expected")
# print("3")
# print("5")
# print("2")
# print("0")

# ===================================================================================================== #
# Method 2: Classification style
def desired_output(input_list):
    if sum(input_list) < 6:
        return sum(input_list)
    else:
        return 11 - sum(input_list)

num_inputs = 11
num_outputs = 6

num_train_exemplars = 5000
num_test_exemplars = 100

# create training and test exemplars
x_train = []
y_train = []
x_test = []
y_test = []

# creating training exemplars
for i in range(num_train_exemplars):
    chance = np.random.uniform(0.0, 1.0)
    x_train.append(np.random.choice([0, 1], size=(11,), p=[chance, 1 - chance]))
    y_train.append(desired_output(x_train[i]))
x_train = np.array(x_train)
y_train = np.array(y_train)
print('training exemplar created')

# creating testing exemplars
for i in range(num_test_exemplars):
    chance = np.random.uniform(0.0, 1.0)
    x_test.append(np.random.choice([0, 1], size=(11,), p=[chance, 1 - chance]))
    y_test.append(desired_output(x_test[i]))
x_test = np.array(x_test)
y_test = np.array(y_test)
print('testing exemplar created')

# convert class labels (indices) to desired output vectors. For example, for an input from class 1,
# we would like the network output to be [0, 1, 0, 0, 0, 0]
# we have 6 class in this case: 0, 1, 2, 3, 4, 5
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

input_shape = (num_inputs,)
batch_size = 10
num_epochs = 100

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# build the neural network model
model = Sequential()
model.add(Dense(200, activation='relu', name='hidden'))
model.add(Dense(num_outputs, activation='linear', name='output'))
model.compile(loss='mse', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

# fit the model to the dataset
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=2, validation_data=(x_test, y_test))

# evaluate the model
final_train_loss, final_train_accuracy = model.evaluate(x_train, y_train, verbose=0)
final_test_loss, final_test_accuracy = model.evaluate(x_test, y_test, verbose=0)

print("predicted")
print(model.predict([[0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0]]))
print(model.predict([[1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0]]))
print(model.predict([[1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1]]))
print(model.predict([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))

print("expected")
print("[0, 0, 0, 1, 0, 0] = 3")
print("[0, 0, 0, 0, 0, 1] = 5")
print("[0, 0, 1, 0, 0, 0] = 2")
print("[1, 0, 0, 0, 0, 0] = 0")


