# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 13:06:02 2017

@author: Xiaofei
"""

import AlphaConstant
import AlphaUtils
import numpy as np
import matplotlib.pyplot as plt
import datetime
from matplotlib.dates import date2num
    
class StockPrice:

    def __init__(self):
        self.alpha = AlphaUtils.Alpha()
        self.f = AlphaConstant.Function()
        self.i = AlphaConstant.Interval()
        
    def run(self,symbol):
        data = self.getData_100(symbol)
        self.plot(data)

    def getData_100(self,s):
        data = self.alpha.GetData_100(function=self.f.Intraday,symbol=s) 
        return data
    
    def plot(self,data): 
        fig = plt.figure()
        graph = fig.add_subplot(111)
        
        xd = [datetime.datetime.strptime(value,"%Y-%m-%d %H:%M:%S") for (value) in data["timestamp"]]
        x = [date2num(value) for (value) in xd]
        
        plt.plot(x,data["open"])
        graph.set_xticks(x)
        graph.set_xticklabels([value.strftime("%H:%M:%S") for (value) in xd],rotation=90)
        plt.show()
        
if __name__ == "__main__":
    test = StockPrice()
    test.run("BABA")