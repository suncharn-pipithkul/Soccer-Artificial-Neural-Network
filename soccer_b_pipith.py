import pickle  # data file

# libraries for ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

# libraries for mathematics, graphs
import numpy as np
import matplotlib.pyplot as pyplot
import math

# global vars
activation = 'sigmoid'


# helper functions
def scale_down(y, ymin, ymax):
    return (y - ymin) / (ymax - ymin)


def scale_dataset_down(y_list):
    for exemplar in y_list:
        exemplar[0] = scale_down(exemplar[0], min_score, max_score)
        exemplar[1] = scale_down(exemplar[1], min_conceded, max_conceded)
        exemplar[2] = scale_down(exemplar[2], min_drink, max_drink)
        exemplar[3] = scale_down(exemplar[3], min_earn, max_earn)


def scale_up(y, ymin, ymax):
    return y * (ymax - ymin) + ymin

def scale_single_up(y):
    y[0] = scale_up(y[0], min_score, max_score)
    y[1] = scale_up(y[1], min_conceded, max_conceded)
    y[2] = scale_up(y[2], min_drink, max_drink)
    y[3] = scale_up(y[3], min_earn, max_earn)
    return y


def scale_dataset_up(y_list):
    for exemplar in y_list:
        exemplar[0] = scale_up(exemplar[0], min_score, max_score)
        exemplar[1] = scale_up(exemplar[1], min_conceded, max_conceded)
        exemplar[2] = scale_up(exemplar[2], min_drink, max_drink)
        exemplar[3] = scale_up(exemplar[3], min_earn, max_earn)


# input = binary vectors of 22 players.
# For example, [0, 1, 1, 0...] indicates player number 1, 2 will be playing
# while player number 0, 3 will not.
# output = real value vectors
# [# of goals scored, # of goals conceded, # of drinks, money earn]

# read dataset
with open('soccer_data.pickle', 'rb') as f:
    (x_train, y_train), (x_test, y_test) = pickle.load(f)

# calculate number of inputs, outputs, exemplars
num_inputs = x_train.shape[1]  # 22
num_outputs = y_train.shape[1]  # 4

num_train_exemplars = x_train.shape[0]  # 4000
num_test_exemplars = x_test.shape[0]  # 1000

# scaling outputs using min-max scaling
y_all = np.concatenate((y_train, y_test))
max_score, max_conceded, max_drink, max_earn = np.max(y_all, axis=0)
min_score, min_conceded, min_drink, min_earn = np.min(y_all, axis=0)

# scale all y down to [0, 1]
scale_dataset_down(y_train)
scale_dataset_down(y_test)

# training variables
input_shape = (num_inputs,)
batch_size = 10
num_epochs = 100

# build the artificial neural network model
model = Sequential()
model.add(Dense(200, activation='relu', name='hidden'))
model.add(Dense(num_outputs, activation=activation, name='output'))  # output layer
model.compile(loss='mse', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

# fit the model to the dataset
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, verbose=2,
                    validation_data=(x_test, y_test))

# evaluate the model
final_train_loss, final_train_accuracy = model.evaluate(x_train, y_train, verbose=0)
final_test_loss, final_test_accuracy = model.evaluate(x_test, y_test, verbose=0)

# calculate mse for each output
mse_score, mse_conceded, mse_drink, mse_earn = (0, 0, 0, 0)
std_score, std_conceded, std_drink, std_earn = (0, 0, 0, 0)
y_predict = model.predict(x_test)
n = num_test_exemplars

# scale them all back up
scale_dataset_up(y_predict)
scale_dataset_up(y_test)

# calculate mse/ std
for predict, expected in zip(y_predict, y_test):
    mse_score += (expected[0] - round(predict[0])) ** 2
    mse_conceded += (expected[1] - round(predict[1])) ** 2
    mse_drink += (expected[2] - predict[2]) ** 2
    mse_earn += (expected[3] - predict[3]) ** 2
mse_score /= n
mse_conceded /= n
mse_drink /= n
mse_earn /= n

std_score = math.sqrt(mse_score)
std_conceded = math.sqrt(mse_conceded)
std_drink = math.sqrt(mse_drink)
std_earn = math.sqrt(mse_earn)

# printing the evaluation results
print()
print('number of epochs: ' + str(num_epochs))
print('activation function: ' + activation)
print('Final training loss (mean square error): ', final_train_loss)
print('Final test loss (mean square error): ', final_test_loss)
print('std of number of goals scored: ', std_score)
print('std of number of goals conceded: ', std_conceded)
print('std of total # of drinks the team had after the match: ', std_drink)
print('std of money that the team gain/lost through the match: ', std_earn)

# the network structure is input layer, 1 hidden layer, 1 output layer
# there are:
#  - 22 neurons in the input layer
#  - 20 neurons in the hidden layer
#  - 4 neurons in the output layer
# I chose this structure because the additional layer allows the model
# to create a more complex function. Previous model in soccer_a_pipith.py can only create
# a linear separation function.
# In addition, the model in soccer_b_pipith.py perform better in all metrics than soccer_a_pipith.py

# Example of 1 run of both models
# For soccer_a_pipith.py
# number of epochs: 100
# activation function: sigmoid
# Final training loss (mean square error):  0.008967390283942223
# Final test loss (mean square error):  0.00901351124048233
# std of number of goals scored:  0.9848857801796105
# std of number of goals conceded:  1.0765686229869418
# std of total # of drinks the team had after the match:  20.16307086165163
# std of money that the team gain/lost through the match:  244082.63276783406

# for soccer_b_pipith.py
# number of epochs: 100
# activation function: sigmoid
# Final training loss (mean square error):  0.002118448494002223
# Final test loss (mean square error):  0.0021461169235408306
# std of number of goals scored:  0.5385164807134504
# std of number of goals conceded:  0.5744562646538028
# std of total # of drinks the team had after the match:  9.392868779018913
# std of money that the team gain/lost through the match:  122644.36379484097

# model b have lower mse for both training and testing which results in
# much lower std for all 4 predicted variables.

