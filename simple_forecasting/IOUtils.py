#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 12:43:03 2017

@author: xqiu
"""

def GetApi():
    import json
    with open('APIKey.json') as data_file:    
        data = json.load(data_file)
    return data["APIKey"];