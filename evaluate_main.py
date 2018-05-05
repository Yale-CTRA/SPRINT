#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 20:27:52 2018

@author: aditya
"""

import sys
sys.path.append('C:\\Users\\adityabiswas\\Documents\\ML Projects\\SPRINT Code\\third')

from preprocess_main import getData
import evaluate_func as E



def retrieveData():


    df, info = getData()
    split = [0.7, 0, 0.3]
    
    info['id'] = info.pop('ID')
    #info['x'] = ['age', 'sbp', 'screat'] #info.pop('predictors')
    info['x'] = info.pop('predictors')
    info['y'] = info.pop('target')
    info['t'] = info.pop('time')
    info['a'] = info.pop('action')
    
    data = E.Data(df, split, info)
    return data






