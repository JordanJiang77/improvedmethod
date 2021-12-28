import numpy as np
from dataUSE import Datadir0
from dataUSE import dataTarget
from tensorflow.python.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math

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
test1 = dataset[train_size1:]
# print(train1)
# print(test1)

x = list_split(train, 33)
y = list_split(train1, 33)
X_test = list_split(test, 33)
Y_test = list_split(test1, 33)

# train the model and realize test
x = np.array(x)
y = np.array(y)
x_train = np.reshape(x, (x.shape[0], x.shape[1], 1))
y_train = np.reshape(y, (y.shape[0], y.shape[1], 1))
model = models.Sequential()
model.add(layers.LSTM(100, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1))
model.add(layers.Activation('relu'))
model.compile(loss="mse", optimizer="rmsprop", metrics=['accuracy'])
model.fit(x_train, y_train, epochs=200, batch_size=10)

X_test = np.array(X_test)
Y_test = np.array(Y_test)
x_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y_test = np.reshape(Y_test, (Y_test.shape[0], Y_test.shape[1], 1))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test accuracy:', score[1])

predict_y = model.predict(x_test)
predict_y = np.reshape(predict_y, (predict_y.size,))

Valprediction = []
Valtrue = []
lenvalue1 = len(predict_y)
lenvalue2 = len(test)
for i in range(1, lenvalue1, 33):
    Valprediction.append(predict_y[i])
# print(Valprediction)
for j in range(1, lenvalue2, 33):
    Valtrue.append(test[j])
predict_y = predict_y[0:33]
# print(predict_y)
# print(loss)
test = test[0:33]
# print RMSE of predict data and true
testScore = return_rmse(test, predict_y)
print('RMSE'.format(testScore))

# plot the data obtained by one sensor
fig1 = plt.figure(1)
plt.plot(Valprediction, 'r')
plt.plot(Valtrue, 'g')
plt.xlabel("the sensor id")
plt.ylabel("traffic flow")
plt.title('traffic data of one sensor at different time')
plt.show()

# different sensor
fig2 = plt.figure(2)
plt.plot(predict_y, 'g')
plt.plot(test, 'r')
plt.xlabel("the sensor id")
plt.ylabel("traffic flow")
plt.title('traffic data of different sensors at one time')
plt.legend(['predict data','true'])
plt.show()

