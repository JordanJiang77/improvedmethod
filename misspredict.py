import numpy as np
from dataUSE import Datadir0
from dataUSE import dataTarget

from sklearn.metrics import mean_squared_error
import math

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class Improvedlstm:
    def __init__(self, sigema, char_to_idx, idx_to_char, epochs=200, n_a=16, alpha=0.01, batch_size=33):

        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char
        self.epochs = epochs
        self.n_a = n_a
        self.alpha = alpha
        self.parameters = {}
        self.loss = 0.0
        self.n_x = len(char_to_idx)
        self.n_y = len(idx_to_char)
        self.batch_size = batch_size
        self.sigema = sigema
        self.parameters = {}

        std = (1.0 / np.sqrt(self.batch_size + self.n_a))  # Xavier initialisation

        np.random.seed(1)
        self.parameters["Wf"] = np.random.randn(n_a, n_a + self.n_x)*std
        self.parameters["bf"] = np.zeros((self.n_a, 1))
        self.parameters["Wi"] = np.random.randn(self.n_a, self.n_a + self.n_x)*std
        self.parameters["bi"] = np.zeros((self.n_a, 1))
        self.parameters["Wc"] = np.random.randn(self.n_a, self.n_a + self.n_x)*std
        self.parameters["bc"] = np.zeros((self.n_a, 1))
        self.parameters["Wo"] = np.random.randn(self.n_a, self.n_a + self.n_x)*std
        self.parameters["bo"] = np.zeros((n_a, 1))
        self.parameters["Wy"] = np.random.randn(self.n_y, n_a)*std
        self.parameters["by"] = np.zeros((self.n_y, 1))
        self.parameters['sigema'] = self.sigema

    def lstm_cell_forward(self, xt, sigema1, a_prev, c_prev):
        # Retrieve parameters from "parameters"
        Wf = self.parameters["Wf"]
        bf = self.parameters["bf"]
        Wi = self.parameters["Wi"]
        bi = self.parameters["bi"]
        Wc = self.parameters["Wc"]
        bc = self.parameters["bc"]
        Wo = self.parameters["Wo"]
        bo = self.parameters["bo"]
        Wy = self.parameters["Wy"]
        by = self.parameters["by"]
        sigema = self.parameters["sigema"]

        # Retrieve dimensions from shapes of xt and Wy
        n_x, m = xt.shape
        n_y, n_a = Wy.shape

        # Concatenate a_prev and xt (≈3 lines)
        concat = np.zeros((n_a + n_x, m))
        concat[: n_a, :] = a_prev
        concat[n_a :, :] = xt + sigema1

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

    def lstm_forward(self, x, sigema, a0):
        # Initialize "caches", which will track the list of all the caches
        caches = []

        # Retrieve dimensions from shapes of x and parameters['Wy'] (≈2 lines)
        n_x, m, T_x = x.shape
        n_y, n_a = self.parameters['Wy'].shape

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
            a_next, c_next, yt, cache,  = self.lstm_cell_forward(xt=x[:, :, t], sigema1=sigema[:, :, t], a_prev=a_next, c_prev=c_next)
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

    def compute_loss(self, y_hat, y):
        n_y, m, T_x = y.shape
        for t in range(T_x):
            self.loss += 1/m * np.sum((y[:, :, t], (-y_hat[:, :, t])))
        return self.loss

    def lstm_cell_backward(self, dz, da_next, dc_next, cache):
        # Retrieve information from "cache"
        (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt) = cache

        # Retrieve dimensions from xt's and a_next's shape (≈2 lines)
        n_a, m = a_next.shape

        dWy = np.dot(dz, a_next.T)
        dby = np.sum(dz, axis=1, keepdims=True)

        da_next = np.dot(self.parameters['Wy'].T, dz) + da_next

        # Compute gates related derivatives, you can find their values can be found by looking carefully at equations (7) to (10) (≈4 lines)
        dot = da_next * np.tanh(c_next) * ot * (1 - ot)
        dcct = (dc_next * it + ot * (1 - np.square(np.tanh(c_next))) * it * da_next) * (1 - np.square(cct))
        dit = (dc_next * cct + ot * (1 - np.square(np.tanh(c_next))) * cct * da_next) * it * (1 - it)
        dft = (dc_next * c_prev + ot * (1 - np.square(np.tanh(c_next))) * c_prev * da_next) * ft * (1 - ft)

        # Compute parameters related derivatives. Use equations (11)-(14) (≈8 lines)
        concat = np.vstack((a_prev, xt)).T
        dWf = np.dot(dft, concat)
        dWi = np.dot(dit, concat)
        dWc = np.dot(dcct, concat)
        dWo = np.dot(dot, concat)
        dbf = np.sum(dft, axis=1, keepdims=True)
        dbi = np.sum(dit, axis=1, keepdims=True)
        dbc = np.sum(dcct, axis=1, keepdims=True)
        dbo = np.sum(dot, axis=1, keepdims=True)
        dsigema1 = np.dot(self.parameters['Wi'][:, n_a:].T, dit)
        dsigema2 = np.dot(self.parameters['Wf'][:, n_a:].T, dft)
        dsigema3 = np.dot(self.parameters['Wc'][:, n_a:].T, dcct)
        dsigema4 = np.dot(self.parameters['Wo'][:, n_a:].T, dot)
        dsigema = dsigema1+dsigema2+dsigema3+dsigema4

        # Compute derivatives w.r.t previous hidden state, previous memory state and input. Use equations (15)-(17). (≈3 lines)
        da_prev = np.dot(self.parameters['Wf'][:, :n_a].T, dft) + np.dot(self.parameters['Wi'][:, :n_a].T, dit) + np.dot(self.parameters['Wc'][:, :n_a].T, dcct) + np.dot(self.parameters['Wo'][:, :n_a].T, dot)
        dc_prev = dc_next * ft + ot * (1 - np.square(np.tanh(c_next))) * ft * da_next
        dxt = np.dot(self.parameters['Wf'][:, n_a:].T, dft) + np.dot(self.parameters['Wi'][:, n_a:].T, dit) + np.dot(self.parameters['Wc'][:, n_a:].T, dcct) + np.dot(self.parameters['Wo'][:, n_a:].T, dot)

        # Save gradients in dictionary
        gradients = {"dxt": dxt, "da_next": da_prev, "dc_next": dc_prev, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
                     "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo, "dWy": dWy, "dby": dby, "dsigema": dsigema}

        return gradients

    def lstm_backward(self, y, y_hat, caches):

        # Retrieve values from the first cache (t=1) of caches.
        (caches, x) = caches

        # Retrieve dimensions from da's and x1's shapes (≈2 lines)
        n_x, m, T_x = x.shape
        n_a = self.n_a
        # initialize the gradients with the right sizes (≈12 lines)
        dxt = np.zeros((n_x, m, T_x))
        da0 = np.zeros((n_a, m))
        da_next = np.zeros((n_a, m))
        dc_next = np.zeros((n_a, m))
        dsigema = np.zeros((n_x, m, T_x))
        dWf = np.zeros((n_a, n_a + n_x))
        dWi = np.zeros((n_a, n_a + n_x))
        dWc = np.zeros((n_a, n_a + n_x))
        dWo = np.zeros((n_a, n_a + n_x))
        dWy = np.zeros((self.n_y, n_a))
        dbf = np.zeros((n_a, 1))
        dbi = np.zeros((n_a, 1))
        dbc = np.zeros((n_a, 1))
        dbo = np.zeros((n_a, 1))
        dby = np.zeros((self.n_y, 1))
        dz = y_hat - y  # y_hat=softmax(z), dz=dl/dy_hat * dy_hat/dz

        # loop back over the whole sequence
        for t in reversed(range(T_x)):
            # Compute all gradients using lstm_cell_backward
            gradients = self.lstm_cell_backward(dz=dz[:, :, t], da_next=da_next, dc_next=dc_next, cache=caches[t])
            # Store or add the gradient to the parameters' previous step's gradient
            dsigema[:, :, t] = gradients["dsigema"]
            dxt[:, :, t] = gradients["dxt"]
            dWf = dWf+gradients["dWf"]
            dWi = dWi+gradients["dWi"]
            dWc = dWc+gradients["dWc"]
            dWo = dWo+gradients["dWo"]
            dWy = dWy+gradients["dWy"]
            dbf = dbf+gradients["dbf"]
            dbi = dbi+gradients["dbi"]
            dbc = dbc+gradients["dbc"]
            dbo = dbo+gradients["dbo"]
            dby = dby+gradients["dby"]
            da_next = gradients['da_next']
            dc_next = gradients['dc_next']

        # Set the first activation's gradient to the backpropagated gradient da_prev.
        da0 = gradients['da_next']

        gradients = {"dx": dxt, "dsigema": dsigema, "da0": da0, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
                     "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo, "dWy": dWy, "dby": dby}

        return y, gradients

    def update_parameters(self, x, gradients):
        n_x, m, T_x = x.shape
        self.parameters['Wf'] += -self.alpha * gradients["dWf"]
        self.parameters['Wi'] += -self.alpha * gradients["dWi"]
        self.parameters['Wc'] += -self.alpha * gradients['dWc']
        self.parameters['Wo'] += -self.alpha * gradients["dWo"]
        self.parameters['Wy'] += -self.alpha * gradients['dWy']

        self.parameters['bf'] += -self.alpha * gradients['dbf']
        self.parameters['bi'] += -self.alpha * gradients['dbi']
        self.parameters['bc'] += -self.alpha * gradients['dbc']
        self.parameters['bo'] += -self.alpha * gradients['dbo']
        self.parameters['by'] += -self.alpha * gradients['dby']
        for t in reversed(range(T_x)):
            self.parameters['sigema'][:, :, t] = self.parameters['sigema'][:, :, t]-self.alpha * gradients["dsigema"][:, :, t]

    def optimize(self, sigema, X, Y):

        loss = []
        for epoch in range(self.epochs):
            a_prev = np.zeros((self.n_a, 33))
            x_batch = X
            y_batch = Y
            sigema = sigema
            a, y_pred, c, caches = self.lstm_forward(x_batch, sigema, a_prev)
            loss1 = self.compute_loss(y_hat=y_pred, y=y_batch)
            loss.append(loss1)
            x, gradients = self.lstm_backward(y_batch, y_pred, caches)
            self.update_parameters(x, gradients)
        return loss, self.parameters