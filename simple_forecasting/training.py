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


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Convolution1D, MaxPooling1D
from keras.callbacks import Callback
import processing as p
import time


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



WINDOW_SIZE = 30
TARGET_TIME = 15
LAG_SIZE = 1
EMB_SIZE = 1
TRAIN_RATIO = 0.8


################################  Retrieve Data #############################################
print('Data loading..')
data, dates = p.get_data('BABA')
dates = [dt.datetime.strptime(d,'%Y-%m-%d %H:%M:%S').date() for d in dates]
#plt.plot(dates, timeseries)

################################  Preprocess Data #############################################
train,test = p.ts_split(data,TRAIN_RATIO)
#train = p.feature_scaling(train)

################################  Training Set #############################################
X_train, Y_train = p.split_into_chunks(train, WINDOW_SIZE, TARGET_TIME, LAG_SIZE, binary=False)
X_train, Y_train = np.array(X_train), np.array(Y_train)

################################  Testing Set #############################################
X_test, Y_test = p.split_into_chunks(test, WINDOW_SIZE, TARGET_TIME, LAG_SIZE, binary=False)
X_test, Y_test = np.array(X_test), np.array(Y_test)
#X_test = p.feature_scaling(X_test)

################################ Model #############################################
print('Building model...')
model = Sequential()
model.add(Dense(512, input_shape = (WINDOW_SIZE, )))
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
#score = model.evaluate(X_test, Y_test, batch_size=128)
#print(score)

################################ Prediction #############################################
ts_test = p.get_ts(test,WINDOW_SIZE)
ts_test = np.array(ts_test)
predicted = model.predict(ts_test)
predicted = predicted+(test[2*TARGET_TIME-1]- predicted[0])


################################ Predict All #############################################
#predicted_all = model.predict(np.reshape(test[:10],(-1,5)))
#predicted_all = predicted_all+(Y_test[0]- predicted_all[0])
try:
    fig = plt.figure()
    #plt.plot(Y_test[:150], color='black') # BLUE - trained RESULT
    #plt.plot(predicted_all, color='blue') # RED - trained PREDICTION
    #plt.plot(test, color='red') # RED - trained PREDICTION
    x_real = np.linspace(1,1000,num=1000)
    plt.plot(x_real[0:len(test)],test[:], color='green') # GREEN - actual RESULT
    plt.plot(x_real[0:len(predicted)]+WINDOW_SIZE+TARGET_TIME-1,predicted[:], color='red') # ORANGE - restored PREDICTION
    #plt.xticks(np.arange(min(x_real), max(x_real)+1, 2.0))
    plt.axis([-50, 800, 174, 181])
    plt.show()
except Exception as e:
    print(str(e))
    
################################ Predict Test #############################################
fig = plt.figure()
ax = fig.add_subplot(111)
xxx = ax.plot(range(0,2))[0]
ax.relim() 
ax.autoscale_view(True,True,True)
fig.canvas.draw()
plt.show(block=False)
xxx.set_xdata(range(2,4))
fig.canvas.draw()
for i in range (WINDOW_SIZE-1,len(test)):
    try:
        #plt.plot(Y_test[:150], color='black') # BLUE - trained RESULT
        #plt.plot(predicted_all, color='blue') # RED - trained PREDICTION
        #plt.plot(test, color='red') # RED - trained PREDICTION
        plt.plot(x_real[0:i],test[0:i], color='green') # GREEN - actual RESULT
        plt.plot(x_real[0:i-WINDOW_SIZE+1]+WINDOW_SIZE+TARGET_TIME-1,predicted[0:i-WINDOW_SIZE+1], color='red') # ORANGE - restored PREDICTION
        #plt.xticks(np.arange(min(x_real), max(x_real)+1, 2.0))
        plt.axis([-50, 800, 174, 181])
        time.sleep(1)
    except Exception as e:
        print(str(e))
        

    
profit = (predicted[-1]-test[-1])*100/X_test[-1][-1]
print("score:",profit)