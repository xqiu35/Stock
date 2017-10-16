# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 13:33:58 2017

@author: Xiaofei
"""

def GetApi():
    import json
    with open('APIKey.json') as data_file:    
        data = json.load(data_file)
    return data["APIKey"];