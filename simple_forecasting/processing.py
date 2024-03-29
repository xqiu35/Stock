# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:37:40 2016

@author: Alex
"""

import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pylab as plt
import datetime as dt
import time
import Alpha

def load_snp_returns():
    f = open('table.csv', 'rb').readlines()[1:]
    raw_data = []
    raw_dates = []
    for line in f:
        try:
            open_price = float(line.split(',')[1])
            close_price = float(line.split(',')[4])
            raw_data.append(close_price - open_price)
            raw_dates.append(line.split(',')[0])
        except:
            continue

    return raw_data[::-1], raw_dates[::-1]


def load_snp_close():
    import pandas as pd
    data=pd.read_csv('baba.csv')
    raw_data=data['close']
    raw_dates=data['timestamp']
    raw_data = np.flip(raw_data,0)
    raw_dates = np.flip(raw_dates,0)

    return raw_data, raw_dates

def ts_split(data,pct):
    train = data[0:int(len(data)*pct)]
    test = data[int(len(data)*pct):]
    return train,test

def feature_scaling(data):
    s = StandardScaler()
    data_s = s.fit_transform(data)
    return data_s
    
    

def split_into_chunks(data, train, predict, step, binary=True):
    X, Y = [], []
    #data = np.array(data)
    for i in range(0, len(data),step):
        try:
            x_i = data[i:i+train]
            y_i = data[i+train+predict-1]
            x_i,y_i = np.array(x_i), np.array(y_i)
            # Use it only for daily return time series
            if binary:
                if y_i-x_i[-1] > 0.:
                    y_i = [1.,0]
                else:
                    y_i = [0,1.]  
            else:
                y_i = data[i+train+predict]
                x_i = data[i:i+train]
                #timeseries = np.array(data[i:i+train+predict])
                #if scale: timeseries = preprocessing.scale(timeseries)
                #x_i = timeseries[:-1]
                #y_i = timeseries[-1]
        except:
            break
        X.append(x_i)
        Y.append(y_i)
        
    return X, Y


def shuffle_in_unison(a, b):
    # courtsey http://stackoverflow.com/users/190280/josh-bleecher-snyder
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def create_Xt_Yt(X, y, percentage=0.8,scale=True,binary=False):
    X_test = X[int(len(X) * percentage):]
    Y_test = y[int(len(X) * percentage):]
    
    X_train = X[0:int(len(X) * percentage)]
    Y_train = y[0:int(len(y) * percentage)]
       
    if scale:X_train = preprocessing.scale(X_train)
    if scale and not binary:Y_train = preprocessing.scale(Y_train)
    
    #X_train, Y_train = shuffle_in_unison(X_train, Y_train)

    return X_train, X_test, Y_train, Y_test

def get_data(s):
    alpha = Alpha.Alpha()
    data=alpha.GetData(function='TIME_SERIES_INTRADAY',symbol=s,interval='1min',outputsize='full')
    raw_data=data['close']
    raw_dates=data['timestamp']
    raw_data = np.array(raw_data)
    raw_dates = np.array(raw_dates)
    raw_data = np.flip(raw_data,0)
    raw_dates = np.flip(raw_dates,0)

    return raw_data, raw_dates

def get_ts(data,windowsize):
    X = []
    #data = np.array(data)
    for i in range(0, len(data)-windowsize+1,1):
        try:
            x_i = data[i:i+windowsize]
            x_i = np.array(x_i)
        except:
            break
        X.append(x_i)        
    return X
    

