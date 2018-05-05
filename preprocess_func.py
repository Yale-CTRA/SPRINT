#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 18:06:45 2018

@author: aditya
"""

    

import numpy as np
import pandas as pd
from scipy.stats import skewtest, boxcox
import matplotlib as plt
import seaborn as sns
from copy import copy

import sys
sys.path.append('C:\\Users\\adityabiswas\\Documents\\ML Projects\\SPRINT Code\\third')
from helper import one_hot


isBinary = lambda df, varList: np.array([len(np.unique(df.loc[np.isfinite(df[var].values),var])) <= 2 for var in varList], dtype = np.bool)

    
def visualizeData(df, vars):
    for var in vars:
        plt.pyplot.clf()
        sns.distplot(df.loc[np.isfinite(df[var].values),var], label = var)
        yield None



def fixInputs(df):
    # turn smoking marker into 2 vars (assume missing are former-smokers)
    # never_smoke captures never-smokers (1) vs ever-smokers (2 & 3)
    # still_smoke captures current-smokers (3) vs not-current-smokers (1 & 2)
    df['never_smoke'] = df['smoke_3cat'].values == 1
    df['still_smoke'] = df['smoke_3cat'].values == 3
    
    ##################################
    ### blood pressure calculations
    
    # mean artierial pressure (MAP)
    df['map'] = (df['sbp'] + 2*df['dbp'])/3
    # pulse pressure (PP)
    df['pp'] = df['sbp'] - df['dbp']
    #df['bp_ratio'] = df['dbp']/df['sbp']
    
    
    # clinical cvd implies subclinical cvd
    df.loc[df['sub_clinicalcvd'] == 1, 'sub_subclinicalcvd'] = 1
    
    # scale count variable
    #df['n_agents'] = df['n_agents']/np.max(df['n_agents'].values)
    
    # artifact of way data was recorded: if value was calculated to be less than 2, it was marked nan due to inaccuracy
    df.loc[np.logical_and(df['umalcr'].values == np.nan, df['screat'].values != np.nan), 'umalcr'] = 1.5
    
    ###############################################################################################
    ### lipid calculations
    ### chr = cholesterol, trr = triglycerides, ldl/hdl = low/high density lipoproteins
    ### already checked that trr, chr, hdl are nans at the same time (implying same panel)
    
    df['non_hdl'] = df['chr'] - df['hdl']
    df['ldl'] = df['non_hdl'] - 0.2*df['trr']
    df['trr_hdl'] = df['trr']/df['hdl'] 
    df['chr_hdl'] = df['chr']/df['hdl'] 
    

    ## assume all others are hispanics?
    #df.loc[df['race4'].values == 'OTHER', 'race4'] = 'HISPANIC'
    df = one_hot(df, 'race4')

    return df.drop(columns = ['newsiteid', 'race4', 'sbptertile', 'smoke_3cat', 'race_black', 'sub_senior', 'inclusionfrs',
                              'sub_cvd'] + ['noagents', 'sub_ckd'])
    
    

    
    

def fixTargets(df):
    
    def combineOutcomes(df, name1, name2, nameNew):
        nameMod = lambda mod, names: (mod + '_' + name for name in names)
        event1, event2, eventNew = nameMod('event', [name1, name2, nameNew])
        time1, time2, timeNew = nameMod('t', [name1, name2, nameNew])
        
        # find where these vars are nans for both  (aka not recorded)
        noRecord = np.logical_and(np.isnan(df[event1].values), np.isnan(df[event2].values))
        # fill in nans for the logical_or step later to work
        df[[event1, event2]] = df[[event1, event2]].fillna(value = 0)
        # fill large values for nans because we will use a min function later
        df[[time1, time2]] = df[[time1, time2]].fillna(value = 1e8)
        
        # combine into 1 var
        df[eventNew] = np.logical_or(df[event1].values, df[event2].values)
        df[timeNew] = np.minimum(df[time1].values, df[time2].values)
        
        # fix nans for where there is no record
        df.loc[noRecord, [eventNew, timeNew]] = np.nan
        df.drop(columns = [event1, event2, time1, time2], inplace = True)
        return df
        
    # combine vars
    df = combineOutcomes(df, 'albuminuria_ckd', 'albuminuria_nockd', 'albuminurea')
    df = combineOutcomes(df, '50percentreduction_egfr', '30percentreduction_egfr', 'reduction_egfr')
    df = combineOutcomes(df, 'mi', 'nonmiacs', 'acs')

    
    ## recode event primary
    names = ['acs', 'stroke', 'hf', 'cvddeath', 'death']
    events, times = ['event_' + name for name in names], ['t_' + name for name in names]
    e_primary, t_primary = np.zeros(len(df)), np.zeros(len(df))
    df_events, df_times = df[events].values.astype(np.bool), df[times].values
    for i in range(len(df)):
        if np.any(df_events[i,:-1]).item():
            e_val = 1
            t_val = np.min(df_times[i,df_events[i,:]])
        else:
            e_val = 0
            t_val = np.max(df_times[i,:])
        e_primary[i], t_primary[i] = e_val, t_val
    
    #turn death into non-cvd death
    df.loc[df['event_cvddeath'] == 1, 'event_death'] = 0
    
    outcomes = ['primary', 'acs', 'stroke', 'hf', 'cvddeath',  'death', 'ckdcomposite', 'reduction_egfr', 'dialysis', 'albuminurea']
    event_outcomes = ['event_' + var for var in outcomes]
    timeto_outcomes = ['t_' + var for var in outcomes]
    
    # remove outcomes we arent interested in (ie not above)
    varsRemove = df.columns[['t_' in predictor for predictor in df.columns.values]]
    varsRemove = varsRemove.drop(event_outcomes + timeto_outcomes).tolist()
    df.drop(columns = varsRemove, inplace = True)
    
    ## remove those observed for 0 time for every event
    keep = np.logical_not(np.all(df[timeto_outcomes[:7]] == 0, axis = 1))
    df = df[keep]
    
#    ## fix the following weird quirk:
#    ## patients who experience cardiovascular death have their time-to-event for all other events coded as 0
#    timedf = np.zeros((len(df), len(timeto_outcomes)))
#    maxObservedTimes = df[timeto_outcomes].apply(lambda row: np.max(row[np.isfinite(row)]), axis = 1).values
#    def replaceMax(row, maxVal):
#        loc = np.logical_or(row == 0, np.isnan(row))
#        row[loc] = maxVal
#        return row
#    
#    subData = df.loc[:, timeto_outcomes].values.astype('float32')
#    for i in range(len(timedf)):
#        timedf[i,:] = replaceMax(subData[i,:], maxObservedTimes[i])
#        
#    df.loc[:,timeto_outcomes] = timedf
#    df.loc[:,event_outcomes] = df.loc[:,event_outcomes].replace(np.nan, 0)
    
    ## remove outcomes with too few events (dialysis: ~10, ckdcomposite ~20)
#    event_outcomes.remove('event_dialysis')
#    event_outcomes.remove('event_ckdcomposite')
#    timeto_outcomes.remove('t_dialysis')
#    timeto_outcomes.remove('t_ckdcomposite')

    return df, event_outcomes, timeto_outcomes




def isSkewed(df, vars, significance = 1e-5):
    results = np.zeros(len(vars), dtype = np.bool)
    for i, var in enumerate(vars):
        data = df.loc[np.isfinite(df[var]),var].values
        results[i] = skewtest(data)[1] < significance
    return results


def logTrans(df, names):
    for name in names:
        restrictedColumn = df.loc[np.isfinite(df[name]), name].values
        df.loc[np.isfinite(df[name]),name] =  np.log(restrictedColumn + 1)
    return df

def boxcoxTrans(df, names):
    for name in names:
        restriction = np.isfinite(df[name].values)
        restrictedColumn = df.loc[restriction, name].values
        df.loc[restriction,name] =  boxcox(restrictedColumn - np.min(restrictedColumn) + np.e)[0]
    return df


def nanScaler(df, vars):
    m, n = len(df), len(vars)
    scaledMatrix = np.empty((m,n), dtype = np.float32)
    scaledMatrix.fill(np.nan)
    for i, var in enumerate(vars):
        restriction = np.isfinite(df[var].values)
        x = df.loc[restriction,var].values
        mu, sigma = np.mean(x), np.std(x)
        scaledMatrix[restriction,i] = (x - mu)/sigma
    df[vars] = scaledMatrix
    return df
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    