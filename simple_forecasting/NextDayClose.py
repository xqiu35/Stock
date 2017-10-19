# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:47:27 2016

@author: Alex
"""

import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, classification_report
import matplotlib.pylab as plt
import datetime as dt
import time

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Convolution1D, MaxPooling1D
from keras.callbacks import Callback
import processing as p


class TrainingHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy = []
        self.predictions = []
        self.i = 0
        self.save_every = 5000

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('acc'))
        self.i += 1        
        if self.i % self.save_every == 0:        
            pred = model.predict(X_train)
            self.predictions.append(pred)



TRAIN_SIZE = 30
TARGET_TIME = 1
LAG_SIZE = 1
EMB_SIZE = 1

print('Data loading..')
timeseries, dates = p.get_data('AMD')
dates = [dt.datetime.strptime(d,'%Y-%m-%d').date() for d in dates]
#plt.plot(dates, timeseries)

X, Y = p.split_into_chunks(timeseries, TRAIN_SIZE, TARGET_TIME, LAG_SIZE, binary=False)
X, Y = np.array(X), np.array(Y)
X_train, X_test, Y_train, Y_test = p.create_Xt_Yt(X, Y, percentage=0.9,scale=True)

Xp, Yp = p.split_into_chunks(timeseries, TRAIN_SIZE, TARGET_TIME, LAG_SIZE, binary=False)
Xp, Yp = np.array(Xp), np.array(Yp)
X_trainp, X_testp, Y_trainp, Y_testp = p.create_Xt_Yt(Xp, Yp, percentage=0.9,scale=False)


print('Building model...')
model = Sequential()
model.add(Dense(512, input_shape = (TRAIN_SIZE, )))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('linear'))
model.compile(optimizer='adam', 
              loss='mse')

model.fit(X_train, 
          Y_train, 
          nb_epoch=8, 
          batch_size = 128, 
          verbose=1, 
          validation_split=0.1)
score = model.evaluate(X_test, Y_test, batch_size=64)
print(score)


params = []
for xt in Y_trainp:
    xt = np.array(xt)
    mean_ = xt.mean()
    scale_ = xt.std()
    params.append([mean_, scale_])

predicted = model.predict(X_test)
new_predicted = []

for pred, par in zip(predicted, params):
    a = pred*par[1]
    a += par[0]
    new_predicted.append(a)
    

mse = mean_squared_error(predicted, new_predicted)
print(mse)

predicted = predicted+(Y_testp[0]- predicted[0])
new_predicted = new_predicted+((Y_testp[0]- new_predicted[0]))

try:
    fig = plt.figure()
    #plt.plot(Y_test[:150], color='black') # BLUE - trained RESULT
    #plt.plot(predicted[:150], color='blue') # RED - trained PREDICTION
    plt.plot(Y_testp[:150], color='green') # GREEN - actual RESULT
    plt.plot(new_predicted[:150], color='red') # ORANGE - restored PREDICTION
    plt.show()
except Exception as e:
    print(str(e))
    
profit = (predicted[-1]-X_testp[-1][-1])*100/X_testp[-1][-1]
print("score:",profit)