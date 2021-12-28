import numpy as np
from dataUSE import Datadir0
from lossMiss import parame
import matplotlib.pyplot as plt
from random import uniform


def list_split(items, n):
    return [items[i:i+n] for i in range(0, len(items), n)]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def lstm_cell_forward(Wy, Wf, Wi, Wc, Wo, by, bf, bi, bc, bo, xt, a_prev, c_prev):
    # Retrieve dimensions from shapes of xt and Wy
    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    # Concatenate a_prev and xt (≈3 lines)
    concat = np.zeros((n_a + n_y, m))
    concat[: n_a, :] = a_prev
    concat[n_a :, :] = xt

    # Compute values for ft, it, cct, c_next, ot, a_next using the formulas given figure (4) (≈6 lines)
    ft = sigmoid(np.dot(Wf, concat) + bf)
    it = sigmoid(np.dot(Wi, concat) + bi)
    cct = np.tanh(np.dot(Wc, concat) + bc)
    c_next = np.multiply(ft, c_prev) + np.multiply(it, cct)
    ot = sigmoid(np.dot(Wo, concat) + bo)
    a_next = np.multiply(ot, np.tanh(c_next))

    # Compute prediction of the LSTM cell (≈1 line)
    yt_pred = np.dot(Wy, a_next) + by

    # store values needed for backward propagation in cache
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt)

    return a_next, c_next, yt_pred, cache

def lstm_forward(Wy, x, a0):
    # Initialize "caches", which will track the list of all the caches
    caches = []

    # Retrieve dimensions from shapes of x and parameters['Wy'] (≈2 lines)
    n_x, m, T_x = x.shape
    n_y, n_a = Wy.shape

    # initialize "a", "c" and "y" with zeros (≈3 lines)
    a = np.zeros((n_a, m, T_x))
    c = np.zeros((n_a, m, T_x))
    y = np.zeros((n_y, m, T_x))

    # Initialize a_next and c_next (≈2 lines)
    a_next = a0
    c_next = np.zeros((n_a, m))

    # loop over all time-steps
    for t in range(T_x):
        # Update next hidden state, next memory state, compute the prediction, get the cache (≈1 line)
        a_next, c_next, yt, cache,  = lstm_cell_forward(Wy, Wf, Wi, Wc, Wo, by, bf, bi, bc, bo, xt=x[:, :, t], a_prev=a_next, c_prev=c_next)
        # Save the value of the new "next" hidden state in a (≈1 line)
        a[:, :, t] = a_next
        # Save the value of the prediction in y (≈1 line)
        y[:, :, t] = yt
        # Save the value of the next cell state (≈1 line)
        c[:, :, t] = c_next
        # Append the cache into caches (≈1 line)
        caches.append(cache)

    # store values needed for backward propagation in cache
    caches = (caches, x)

    return a, y, c, caches

def compute_loss(y_hat, y):
    n_y, m, T_x = y.shape
    loss = 0
    for t in range(T_x):
        loss += 1/m * np.sum((y[:, :, t], (-y_hat[:, :, t])))
    return loss

Wy = parame["Wy"]
Wf = parame["Wf"]
Wi = parame["Wi"]
Wc = parame["Wc"]
Wo = parame["Wo"]
by = parame["by"]
bf = parame["bf"]
bi = parame["bi"]
bc = parame["bc"]
bo = parame["bo"]

# fix random seed for reproducibility
np.random.seed(7)

# load data
dataframe = Datadir0
dataset = dataframe
# print(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.75)
train = dataset[0:train_size]
test = dataset[train_size:]
print(train)
# print(test)

t = list_split(test, 33)
t = np.array(t)
Testbatch = np.reshape(t, (t.shape[0], t.shape[1], 1))
n_x, m, T_x = Testbatch.shape
a_prev = np.zeros((16, 33))
c_next = np.zeros((16, m))

for t in range(T_x):
    a_next, c_next, yt, cache,  = lstm_cell_forward(Wy, Wf, Wi, Wc, Wo, by, bf, bi, bc, bo, xt=Testbatch[:, :, t], a_prev=a_prev, c_prev=c_next)
    # print(yt)
x_batch = Testbatch
y_batch = Testbatch
a, y_pred, c, caches = lstm_forward(Wy, x_batch, a_prev)
loss1 = compute_loss(y_hat=y_pred, y=y_batch)
print(loss1)