
import pandas_datareader as web

df1 = web.DataReader("GOOGL", data_source='yahoo', start='2012-01-01')

n1 = df1.shape[0]

import math

data1 = df1.filter(["Close"])
dataset1 = data1.values
tr_dat_len1 = math.ceil(len(dataset1) * 0.8)
tr_dat_len1

from sklearn.preprocessing import MinMaxScaler

sc1 = MinMaxScaler(feature_range=(0, 1))
sc_data1 = sc1.fit_transform(dataset1)
sc_data1

train_data1 = sc_data1[0:tr_dat_len1, :]
x_train1 = []
y_train1 = []
for i1 in range(60, len(train_data1)):
    x_train1.append(train_data1[i1 - 60:i1, 0])
    y_train1.append(train_data1[i1, 0])
    if i1 <= 60:
        print("x_train:", x_train1)
        print()
        print("y_train:", y_train1)

import numpy as np

x_train1, y_train1 = np.array(x_train1), np.array(y_train1)
x_train1 = np.reshape(x_train1, (x_train1.shape[0], x_train1.shape[1], 1))
x_train1.shape

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

model1 = Sequential()
model1.add(LSTM(50, return_sequences=True))
model1.add(LSTM(50, return_sequences=False))
model1.add(Dense(25))
model1.add(Dense(1))

model1.compile(optimizer='adam', loss='mean_squared_error')

model1.fit(x_train1, y_train1, batch_size=1, epochs=1)

df2 = web.DataReader("AAPL", data_source='yahoo', start='2012-01-01')

n2 = df2.shape[0]

import math

data2 = df2.filter(["Close"])
dataset2 = data2.values
tr_dat_len2 = math.ceil(len(dataset2) * 0.8)
tr_dat_len2

from sklearn.preprocessing import MinMaxScaler

sc2 = MinMaxScaler(feature_range=(0, 1))
sc_data2 = sc2.fit_transform(dataset2)
sc_data2

train_data2 = sc_data2[0:tr_dat_len2, :]
x_train2 = []
y_train2 = []
for i2 in range(60, len(train_data2)):
    x_train2.append(train_data2[i2 - 60:i2, 0])
    y_train2.append(train_data2[i2, 0])
    if i2 <= 60:
        print("x_train:", x_train2)
        print()
        print("y_train:", y_train2)

import numpy as np

x_train2, y_train2 = np.array(x_train2), np.array(y_train2)
x_train2 = np.reshape(x_train2, (x_train2.shape[0], x_train2.shape[1], 1))
x_train2.shape

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

model2 = Sequential()
model2.add(LSTM(50, return_sequences=True))
model2.add(LSTM(50, return_sequences=False))
model2.add(Dense(25))
model2.add(Dense(1))

model2.compile(optimizer='adam', loss='mean_squared_error')

model2.fit(x_train2, y_train2, batch_size=1, epochs=1)

test_data2 = sc_data2[tr_dat_len2 - 60:, :]
x_test2 = []
y_test2 = dataset2[tr_dat_len2:, :]
for i in range(60, len(test_data2)):
    x_test2.append(test_data2[i - 60:i, 0])

x_test2 = np.array(x_test2)
x_test2 = np.reshape(x_test2, (x_test2.shape[0], x_test2.shape[1], 1))

df3 = web.DataReader("GLD", data_source='yahoo', start='2012-01-03')
df3

n3 = df3.shape[0]

import math

data3 = df3.filter(["Close"])
dataset3 = data3.values
tr_dat_len3 = math.ceil(len(dataset3) * 0.8)
tr_dat_len3

from sklearn.preprocessing import MinMaxScaler

sc3 = MinMaxScaler(feature_range=(0, 1))
sc_data3 = sc3.fit_transform(dataset2)

sc_data3

train_data3 = sc_data3[0:tr_dat_len3, :]
x_train3 = []
y_train3 = []

for i3 in range(60, len(train_data3)):
    x_train3.append(train_data2[i3 - 60:i3, 0])
    y_train3.append(train_data2[i3, 0])
    if i3 <= 60:
        print("x_train:", x_train3)
        print()
        print("y_train:", y_train3)

import numpy as np

x_train3, y_train3 = np.array(x_train3), np.array(y_train3)

x_train3 = np.reshape(x_train3, (x_train3.shape[0], x_train3.shape[1], 1))
x_train3.shape

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

model3 = Sequential()
model3.add(LSTM(50, return_sequences=True))
model3.add(LSTM(50, return_sequences=False))
model3.add(Dense(25))
model3.add(Dense(1))

model3.compile(optimizer='adam', loss='mean_squared_error')

model3.fit(x_train3, y_train3, batch_size=1, epochs=1)

df4 = web.DataReader("BTC", data_source='yahoo', start='2020-10-01')
df4

n4 = df4.shape[0]

import math

data4 = df4.filter(["Close"])
dataset4 = data4.values
tr_dat_len4 = math.ceil(len(dataset4) * 0.8)
tr_dat_len4

from sklearn.preprocessing import MinMaxScaler

sc4 = MinMaxScaler(feature_range=(0, 1))
sc_data4 = sc4.fit_transform(dataset4)

sc_data4

train_data4 = sc_data4[0:tr_dat_len4, :]
x_train4 = []
y_train4 = []

for i4 in range(60, len(train_data4)):
    x_train4.append(train_data4[i4 - 60:i4, 0])
    y_train4.append(train_data4[i4, 0])
    if i4 <= 60:
        print("x_train:", x_train4)
        print()
        print("y_train:", y_train4)

import numpy as np

x_train4, y_train4 = np.array(x_train4), np.array(y_train4)

x_train4 = np.reshape(x_train4, (x_train4.shape[0], x_train4.shape[1], 1))
x_train4.shape

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

model4 = Sequential()
model4.add(LSTM(50, return_sequences=True))
model4.add(LSTM(50, return_sequences=False))
model4.add(Dense(25))
model4.add(Dense(1))

model4.compile(optimizer='adam', loss='mean_squared_error')

model4.fit(x_train4, y_train4, batch_size=1, epochs=1)

df5 = web.DataReader("DOGE-USD", data_source='yahoo', start='2017-11-09')
df5

n5 = df5.shape[0]

import math

data5 = df5.filter(["Close"])
dataset5 = data5.values
tr_dat_len5 = math.ceil(len(dataset5) * 0.8)
tr_dat_len5

from sklearn.preprocessing import MinMaxScaler

sc5 = MinMaxScaler(feature_range=(0, 1))
sc_data5 = sc5.fit_transform(dataset5)

sc_data5 = dataset5

train_data5 = sc_data5[0:tr_dat_len5, :]
x_train5 = []
y_train5 = []

for i5 in range(60, len(train_data5)):
    x_train5.append(train_data5[i5 - 60:i5, 0])
    y_train5.append(train_data5[i5, 0])
    if i5 <= 60:
        print("x_train:", x_train5)
        print()
        print("y_train:", y_train5)

import numpy as np

x_train5, y_train5 = np.array(x_train5), np.array(y_train5)

x_train5 = np.reshape(x_train5, (x_train5.shape[0], x_train5.shape[1], 1))
x_train5.shape

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

model5 = Sequential()
model5.add(LSTM(50, return_sequences=True))
model5.add(LSTM(50, return_sequences=False))
model5.add(Dense(25))
model5.add(Dense(1))

model5.compile(optimizer='adam', loss='mean_squared_error')

model5.fit(x_train5, y_train5, batch_size=1, epochs=1)

go_quote1 = web.DataReader('DOGE-USD', data_source='yahoo', start='2012-01-01', end='2022-06-06')
ndf1 = go_quote1.filter(['Close'])
l60d1 = ndf1[-60:].values
l60d_sc1 = sc1.transform(l60d1)
X_test1 = []
X_test1.append(l60d_sc1)
X_test1 = np.array(X_test1)
X_test1 = np.reshape(X_test1, (X_test1.shape[0], X_test1.shape[1], 1))
pred_price1 = model5.predict(X_test1)
pred_price1 = sc1.inverse_transform(pred_price1)
pr = pred_price1
pr

fn1 = 'model1.h5'
model1.save(fn1)

fn2 = 'model2.h5'
model2.save(fn2)

fn3 = 'model3.h5'
model3.save(fn3)

fn4 = 'model4.h5'
model4.save(fn4)

fn5 = 'model5.h5'
model5.save(fn5)