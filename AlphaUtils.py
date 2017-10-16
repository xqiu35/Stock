# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 13:33:58 2017

@author: Xiaofei
"""

class Alpha:
    def __init__(self):
        import IOUtils
        self.function=""
        self.symbol=""
        self.interval=""
        self.outputsize="compact" # or compact
        self.datatype="csv" #or csv
        self.apikey=IOUtils.GetApi()
        self.url="https://www.alphavantage.co/query"
        self.vars={}

    def loadVars(self):
        self.vars['function'] = self.function
        self.vars['symbol'] = self.symbol
        self.vars['interval'] = self.interval
        self.vars['outputsize'] = self.outputsize
        self.vars['datatype'] = self.datatype
        self.vars['apikey'] = self.apikey
        self.vars['url'] = self.url
        
    def GetData_100(self,function,symbol,interval="1min",outputsize="compact"):
        import requests
        self.function = function
        self.symbol = symbol
        self.interval = interval
        self.datatype = "csv"
        self.outputsize = outputsize
        self.loadVars()
        data = requests.get(self.url, params=self.vars).content
        import pandas as pd
        import io
        data=pd.read_csv(io.StringIO(data.decode('utf-8')))
        return data;
    
    def GetData_1(self,function,symbol,interval="1min",outputsize="compact"):
        import requests
        self.function = function
        self.symbol = symbol
        self.interval = interval
        self.datatype = "json"
        self.outputsize = outputsize
        self.loadVars()
        data = requests.get(self.url, params=self.vars)
        return data;
    
        
        
        