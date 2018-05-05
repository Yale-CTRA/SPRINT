#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:10:09 2018

@author: aditya
"""

import numpy as np
import pandas as pd
import os
import sys
from copy import copy
from sklearn.preprocessing import Imputer

sys.path.append('C:\\Users\\adityabiswas\\Documents\\ML Projects\\SPRINT Code\\third')
import preprocess_func as G


def getData():
    os.chdir('C:\\Users\\adityabiswas\\Documents\\ML Projects\\SPRINT Data')
    dataFileName = 'baseline and all events corrected.dta'
    
    df = pd.read_stata(dataFileName)
    ID = 'maskid'
    df.set_index(ID, drop = False, inplace = True)
    idIndex = df.columns.get_loc(ID)
    idVec = np.zeros(len(df))
    for i in range(len(df)):
        idVec[i] = df.iloc[i,idIndex][1:]
    df[ID] = idVec
    
    
    df, eventTargets, timeTargets = G.fixTargets(df)
    df = G.fixInputs(df)
    
    treatment = 'intensive'
    infoVec = [ID, treatment] + eventTargets + timeTargets
    predictors = list(set(df.columns.values) - set(infoVec))
    binaryPredictors = list(np.array(predictors)[G.isBinary(df, predictors)])
    floatPredictors = list(set(predictors) - set(binaryPredictors))
    df = df[[ID, treatment] + binaryPredictors + floatPredictors + eventTargets + timeTargets].astype(np.float32)
        
    
    ## turn on for boxcox transform
    if False:
        floatPredictorsRestricted = copy(floatPredictors)
        floatPredictorsRestricted.remove('n_agents')
        varsTransform = list(np.array(floatPredictorsRestricted)[G.isSkewed(df, floatPredictorsRestricted, significance = 1e-50)])
        df = G.boxcoxTrans(df, varsTransform)
        
    ## turn on for biased scaling and imputing
    if False:
        df = G.nanScaler(df, floatPredictorsRestricted)
    if True:
        imputer = Imputer()
        df[predictors] = imputer.fit_transform(df[predictors].values)
        
    
    ## make sure imputer is true, scaler/boxcox is false for this
    ## round variables if using trees to make line search easier
    ## risk10yrs, bmi, umalcr, egfr || chr_hdl, trr_hdl
    if False:
        roundVars = ['risk10yrs', 'bmi', 'umalcr', 'egfr', 'chr_hdl', 'trr_hdl']
        df[roundVars] = np.round(df[roundVars].values, decimals = 1)
        
    ## removes people censored at 0
    if True:
        df = df.loc[df[timeTargets[0]] > 0,:]

        
    #predictors = ['egfr', 'umalcr', 'screat', 'glur', 'age', 'pp', 'sbp', 'map', 'dbp', 'trr', 'sub_ckd']
    ## some issue  ckdcomposite and dialysis events and times showing up in predictors
#    predictors = list(set(predictors) - set(['black', 'white', 'hispanic', 'other', 'female',
#                                          'aspirin', 'statin', 'still_smoke', 'never_smoke',
#                                          'sub_subclinicalcvd', 'sub_clinicalcvd', 'n_agents',
#                                          'trr_hdl', 'chr_hdl']))
    info = {'ID': ID, 'predictors': predictors, 'action': treatment, 'target': eventTargets[:6], 'time': timeTargets[:6]}
    
    ## reverse Y
    #df[eventTargets] = 1 - df[eventTargets]
    #maxTime = np.max(df['t_primary'].values)
    #df['t_primary'] = df['t_primary'].values/maxTime
    
    return df, info
    
#x = F.visualizeData(df, floatPredictors)
#next(x)