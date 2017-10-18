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
timeseries, dates = p.load_snp_close()
dates = [dt.datetime.strptime(d,'%Y-%m-%d').date() for d in dates]
plt.plot(dates, timeseries)

X, Y = p.split_into_chunks(timeseries, TRAIN_SIZE, TARGET_TIME, LAG_SIZE, binary=True)
X, Y = np.array(X), np.array(Y)
X_train, X_test, Y_train, Y_test = p.create_Xt_Yt(X, Y, percentage=0.9,scale=True,binary=True)

Xp, Yp = p.split_into_chunks(timeseries, TRAIN_SIZE, TARGET_TIME, LAG_SIZE, binary=True)
Xp, Yp = np.array(Xp), np.array(Yp)
X_trainp, X_testp, Y_trainp, Y_testp = p.create_Xt_Yt(Xp, Yp, percentage=0.9,scale=True,binary=True)



model = Sequential()
model.add(Convolution1D(input_shape = (TRAIN_SIZE, EMB_SIZE), 
                        nb_filter=128,
                        filter_length=2,
                        border_mode='same',
                        activation='relu',
                        subsample_length=1))
model.add(MaxPooling1D(pool_length=2))

model.add(Convolution1D(input_shape = (TRAIN_SIZE, EMB_SIZE), 
                        nb_filter=128,
                        filter_length=2,
                        border_mode='same',
                        activation='relu',
                        subsample_length=1))
model.add(MaxPooling1D(pool_length=2))

model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(250))
model.add(Dropout(0.25))
model.add(Activation('relu'))
model.add(Dense(2))
model.add(Activation('softmax'))

history = TrainingHistory()

model.compile(optimizer='adam', 
        			 loss='binary_crossentropy', 
        			 metrics=['accuracy'])


model.fit(X_train[:, :, np.newaxis], Y_train,
          batch_size=2,
          epochs=5,
          validation_data=(X_test[:, :, np.newaxis], Y_test),
                           shuffle=False)


#score = model.evaluate(X_test[:, :, np.newaxis], Y_test, batch_size=2)
#print(score)

predicted = model.predict(X_test[:, :, np.newaxis])
