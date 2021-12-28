import numpy as np
from dataUSE import Datadir0
from dataUSE import dataTarget
from tensorflow.python.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math

from Lstmcla import LSTM
from random import uniform
def list_split(items, n):
    return [items[i:i+n] for i in range(0, len(items), n)]

def return_rmse(test,predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    print("rmse is {}.".format(rmse))

# fix random seed for reproducibility
np.random.seed(7)

# load data
dataframe = Datadir0
dataset = dataframe
# print(dataset)

# load target
targetset = dataTarget

# split into train and test sets
train_size = int(len(dataset) * 0.75)
train = dataset[0:train_size]
test = dataset[train_size:]
print(train)
# print(test)

# target splitting
train_size1 = int(len(dataset) * 0.75)
train1 = dataset[0:train_size1]
test1 = dataset[train_size:]
# print(train1)
# print(test1)

x = list_split(train, 33)
y = list_split(train1, 33)

# train the model and realize test
x = np.array(x)
y = np.array(y)
x_train = np.reshape(x, (x.shape[0], x.shape[1], 1))
y_train = np.reshape(y, (y.shape[0], y.shape[1], 1))

char_to_idx = x_train
idx_to_char = y_train
vocab_size = 33

lstm = LSTM(char_to_idx, idx_to_char)
loss = lstm.optimize(char_to_idx, idx_to_char)
print(loss)
# print(parame)

fig1 = plt.figure(1)
plt.title('the change curve of loss 200 epochs')
for i in range(0,1,199):
    plt.plot(loss, 'r')
    plt.show()
