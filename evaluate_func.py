#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 19:22:02 2018

@author: aditya
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('C:\\Users\\adityabiswas\\Documents\\ML Projects\\SPRINT Code\\third')
from LogRankRF import SurvStats

def makeSetIndices(m, trainPercent, valPercent):
    trainIndex = np.repeat(False, m)
    valIndex = np.repeat(False, m)
    testIndex = np.repeat(False, m)
    
    cutTrain = int(round(m*trainPercent))
    cutVal = int(round(m*(trainPercent + valPercent)))
    
    trainIndex[:cutTrain] = True
    valIndex[cutTrain:cutVal] = True
    testIndex[cutVal:] = True
    
    return trainIndex, valIndex, testIndex



class Data(object):
    def __init__(self, df, split, infoDict):
        assert sum(split) > 0.99 and sum(split) < 1.01
        self.df = df.astype(np.float32)
        self.split = split
        self.infoDict = infoDict
        
        self.refresh()
        
    def refresh(self, seed = None):
        if seed is not None:
            np.random.seed(seed)
        df = self.df.sample(frac = 1.)
        trainPercent, valPercent = self.split[0:2]
        trainIndex, valIndex, testIndex = makeSetIndices(len(df), trainPercent, valPercent)
        self.train, self.val, self.test = {}, {}, {}
        
        for key in self.infoDict:
            self.train[key] = df.loc[trainIndex, self.infoDict[key]].values
            self.val[key] = df.loc[valIndex, self.infoDict[key]].values
            self.test[key] = df.loc[testIndex, self.infoDict[key]].values



def refresh(df, split, info, rowNames = None, colNames = None):
    assert sum(split) > 0.99 and sum(split) < 1.01
    trainPercent, valPercent = split[0], split[1]
    df = df.sample(frac = 1.)
    trainIndex, valIndex, testIndex = makeSetIndices(len(df), trainPercent, valPercent)
    dfTrain, dfVal, dfTest = df[trainIndex], df[valIndex], df[testIndex]
    dfTrain = dfTrain.sort_values(by = 't_primary')
    
    dfDict = {'train' :dfTrain, 'val': dfVal, 'test': dfTest}
    data = separate(dfDict, info, rowNames = rowNames, colNames = colNames)
    return data



def separate(dfDict, info, rowNames = None, colNames = None):
    
    if colNames is None:
        colNames = list(info)
    else:
        assert len(colNames) == len(info)
        
    if rowNames is None:
        rowNames = list(dfDict) 
    else:
        assert len(rowNames) == len(dfDict)

    separateSingleDF = lambda df, info: [df[info[key]].values for key in info]
    
    data = pd.DataFrame(columns = colNames, index = rowNames, dtype = object)
    for i, key in enumerate(dfDict):
        dfSeparated = separateSingleDF(dfDict[key], info)
        for j in range(len(colNames)):
            data.loc[rowNames[i],colNames[j]] = dfSeparated[j]
        
    return data


def AUUC(data, U, bins = 5, graph = True, label = None, verbose = True):
    #combine and sort data
    IDs, Y, A = data.loc['id'], data.loc['y'], data.loc['a']
    sort_index = np.argsort(U)[::-1]
    df = pd.DataFrame({'Uplift': U, 'target': Y, 'treatment': A}, index = IDs).iloc[sort_index,:]
    
    # organize slices
    m = len(df)
    bin_size = int(np.floor(m/bins))
    slices = [slice(0, bin_size*(i+1)) for i in range(bins-1)]
    slices += [slice(0,m)]
    
    data_binned = [df.iloc[locations,:] for locations in slices]
    Y_t = [sum(df['target'][df['treatment']==1]) for df in data_binned]
    Y_c = [sum(df['target'][df['treatment']==0])for df in data_binned]
    N_t = [sum(df['treatment'] == 1) for df in data_binned]
    N_c = [sum(df['treatment'] == 0) for df in data_binned]
    U = [100*(Y_t[i] - Y_c[i]*N_t[i]/N_c[i] + Y_t[i]*N_c[i]/N_t[i] - Y_c[i])/m for i in range(bins)]
    avg_effect = 100*(Y_t[-1]/N_t[-1] - Y_c[-1]/N_c[-1])
    
    X = [0] + [(b+1)*100/bins for b in list(range(bins))]
    Y = [0] + [U[i] for i in range(bins)]
    areas = [1/bins*(U[i]+U[i+1])/2 for i in range(bins-1)]
    auuc = sum(areas) - avg_effect*1/2

    if graph:
        plt.plot(X, Y, label = label)
        plt.plot([0, 100], [0, avg_effect])
        plt.ylabel('% of Pop. Lives Saved')
        plt.xlabel('% of Pop. Targeted')
        plt.legend(loc = 'upper left')
        
    if verbose:
        label = '' if label is None else label
        print('{} AUUC is: {}'.format(label, auuc))
    return(auuc)


def showCoef(coef, predictors):
    if len(np.shape(coef)) > 1:
        coef = coef[0]
    coef = np.around(coef, decimals = 3)
    sortIndex = np.argsort(coef)
    ## make sure coef is 1 dimensional
    lengths = [len(x) for x in predictors]
    k = max(lengths)
    spaces = [k - len(x) for x in predictors]
    predictors = [predictors[i] + ' '*spaces[i] for i in range(len(predictors))]
    
    toPrint = np.array(list((zip(predictors, coef))))
    toPrint = toPrint[sortIndex]
    print('\n')
    for i in range(len(toPrint)):
        print(toPrint[i])
    print('\n')
    

def convertForOutput(data, U):
    uTrain, uVal, uTest = U
    uVal = np.zeros(0) if uVal is None else uVal
    
    def revertID(x):
        m = len(x)
        x = x.astype('int64').astype('<U18')
        new = np.zeros(m, dtype = '<U18')
        for i in range(m):
            new[i] = 'S' + x[i]
        return new
    
    
    df = [np.concatenate((revertID(data.loc[0,'id']), revertID(data.loc[1,'id']), revertID(data.loc[2,'id'])))[:,None]]
    df += [np.concatenate((data.loc[0,'y'], data.loc[1,'y'], data.loc[2,'y']))[:,None]]
    df += [np.concatenate((data.loc[0,'a'], data.loc[1,'a'], data.loc[2,'a']))[:,None]]
    
    training = np.zeros((len(df[0]),1), dtype = np.bool)
    training[:len(data.loc[0,'y']) + len(data.loc[1,'y'])] = True
    df += [training]
    U = np.concatenate((uTrain, uVal, uTest))[:,None]
    df += [U]
    
    names = ['id', 'y', 'a', 'training', 'uplift']
    df = np.concatenate(df, axis = 1)
    df = pd.DataFrame(df, columns = names)
    return df


def getUplift(data, U, bins = 10):
    rowIndex = None
    for rowName in list(data.index):
        if len(U) == len(data.loc[rowName,'y']):
            rowIndex = rowName
    assert rowIndex is not None
    
    data = np.concatenate((data.loc[rowIndex,'y'][:,None], data.loc[rowIndex,'t'][:,None], data.loc[rowIndex,'a'][:,None]), axis = 1)
    cutOffs = np.percentile(U, np.arange(0, 100, bins))
    scoreVec = np.zeros(bins, dtype = np.float32)
    for i in range(bins):
        subsetIndex = U >= cutOffs[i]
        scoreVec[i] = getScore(data[subsetIndex,:])
    return scoreVec



def deltaRec(U, Y, T, A, p, threshold, totalDays):
    ## asssumes time has been scaled such that maxTime = 1
    recIndex = np.logical_or(np.logical_and(U >= threshold, A), np.logical_and(U < threshold, ~A))
    recMedian = KMpercentile(*KMestimate(Y[recIndex], T[recIndex]), p = p)
    antirecMedian = KMpercentile(*KMestimate(Y[~recIndex], T[~recIndex]), p = p)
    diffDays = totalDays*(recMedian - antirecMedian)
    return diffDays


def deltaTreat(U, Y, T, A, p, threshold, totalDays):
    subsetIndex = U >= threshold
    Y, T, A = Y[subsetIndex], T[subsetIndex], A[subsetIndex]
    treatMedian = KMpercentile(*KMestimate(Y[A], T[A]), p = p)
    controlMedian = KMpercentile(*KMestimate(Y[~A], T[~A]), p = p)
    diffDays = totalDays*(treatMedian - controlMedian)
    return diffDays


def setVars(data):
    return data['y'], data['t'], data['a']

def AUDRC(U, Y, T, A, totalDays, p = 0.95, bins = 10, plot = False):
    binSize = 100/bins
    A = A == 1
    decisions = list(np.percentile(U, np.arange(0, 100, binSize)))
    deltaRecList = np.zeros(bins + 1, dtype = np.float32)
    #deltaTreatList = np.zeros(bins + 1, dtype = np.float32)
    for i, threshold in enumerate(decisions):
        # i == 0 is case when rec group is treated and antirec group is control
        deltaRecList[i] = deltaRec(U, Y, T, A, p = p, threshold = threshold, totalDays = totalDays)
        #deltaTreatList[i] = deltaTreat(U, Y, T, A, p = p, threshold = threshold, totalDays = totalDays)
        
    deltaRecList[-1] = -deltaRecList[0]
    #deltaTreatList[-1] = 0
    
    x = np.arange(0, 100 + binSize, binSize)
    if plot:
        plt.figure()
        plt.plot(x, deltaRecList)
       # plt.plot(x, deltaTreatList)
        plt.plot([0, 100], [deltaRecList[0], deltaRecList[-1]])
        
        
    return (np.trapz(deltaRecList + deltaRecList[0], x = x) - deltaRecList[0]*100)/100
        
    

def nanChecker(X):
    if type(X) == pd.core.frame.DataFrame:
        X = X.values
    return np.sum(np.isnan(X)) > 0 



def RMST(Y, T, tau = 365.25*4):
    index = np.lexsort((1-Y, T))
    Y, T = Y[index], T[index]
    
    n = np.arange(len(Y))[::-1] + 1
    eventBool = Y == 1
    eventTimes = T[eventBool]
    times, tIndex, d = np.unique(eventTimes, return_index = True, return_counts = True)
    n = n[eventBool][tIndex]
    
    select = np.logical_and(d > 0, times < tau)
    n, d, times = n[select], d[select], times[select]
    S = np.cumprod(1-d/n)
    deltas = times[1:] - times[:-1]
    area = np.sum(S[:-1]*deltas) + times[0] + (tau-times[-1])*S[-1]
    return area
    
    


def strategyGraph(U, Y, T, A, bins = 10, plot = False, name = None):
    treated = A == 1    
    
    #################################
    ##### CALCULATE RMST #####
    #################################
    
    # Calculate Restricted Mean Survival Times
    treatEventRate = RMST(Y[treated], T[treated])
    controlEventRate = RMST(Y[~treated], T[~treated])
    
    
    # rate in recommended group at different decision boundaries for intelligent treatment strategy
    binSize = 100/bins
    decisionBoundaries = list(np.percentile(U, np.arange(0, 100 + binSize, binSize)) - 1e-5) ## ensure that cutoff is below percentile
    recEventRates = np.zeros(bins + 1, dtype = np.float32)
    for i, threshold, in enumerate(decisionBoundaries):
        treatSelector = U >= decisionBoundaries[i]
        recSelector = np.logical_or(np.logical_and(treatSelector, treated), np.logical_and(~treatSelector, ~treated))
        recEventRates[i] = RMST(Y[recSelector], T[recSelector])
    
    
    ############################
    ##### PLOT and RETURN ######
    ############################
    
    x = np.flip(np.arange(0, 100 + binSize, binSize), axis = 0)
    if plot:            
        fig, ax = plt.subplots()        
        ax.plot(x, recEventRates, 'g', linewidth = 2.5, label = 'Data-Driven Strategy')
        ax.plot([0, 100], [controlEventRate, treatEventRate], 'k', linewidth = 1.5, label = 'Random Strategy')
        ax.plot([0, 100], [treatEventRate, treatEventRate], 'm--', linewidth = 1.5, label = 'Treat-All Strategy')
        ax.plot([0, 100], [controlEventRate, controlEventRate], 'c--', linewidth = 1.5, label = 'Treat-Nobody Strategy')
        legend = ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        for label in legend.get_texts():
            label.set_fontsize('large')
            
        ax.set_xlabel('Percent of Population Targeted', fontsize=18)
        ax.set_ylabel('Cumulative Hazard', fontsize=18)
        ax.set_title(name, fontsize = 18)
        ax.tick_params(labelsize = 14)
        plt.show()
        
    return -(np.trapz([controlEventRate, treatEventRate], [0, 100]) - np.trapz(recEventRates, np.flip(x, axis = 0)))

    


def KMestimate(Y, T, plot = False):
    index = np.argsort(T)
    Y, T = Y[index], T[index]

    times = np.unique(T).tolist()
    m = len(times)
    S = np.ones(m, dtype = np.float32)
    
    for i, t in enumerate(times):
        deathIndex = T == t
        riskIndex = T > t
        riskIndex[np.logical_and(deathIndex, Y == 0)] = True
        z = 1 - np.sum(Y[deathIndex])/np.sum(riskIndex)
        if i is 0:
            S[i] = z
        else:
            S[i] = S[i-1]*z
    
    times = [0] + times
    S = np.concatenate((np.ones(1), S))
    if plot:
        plt.plot(times, S)
    
    return (np.array(times), S)


def recLogRank(results, Y, T, A, cutoff = 0):
    intervention = A == 1
    outcomes = np.stack([Y, T], axis = 1)
    rec = np.logical_or(np.logical_and(results > cutoff, intervention),
                         np.logical_and(results <= cutoff, np.logical_not(intervention)))
    model = SurvStats(outcomes, rec)
    return model.getStatistic()
    


def KMpercentile(times, S, p = 0.95):
    assert p > S[-1]
    idx = np.arange(len(S))[S < p][0]
    return times[idx]


    
def getScore(df):
    Y, T, A = df[:,0], df[:,1], df[:,2]
    A = A == 1
    
    # Y = 1-Y
    times = np.unique(T)
    last = np.argwhere(np.minimum(np.max(T[A]), np.max(T[~A])) == times)[0][0]

    saved = np.zeros(last, dtype = np.float32)
    for i, tau, in enumerate(list(times[:last])):
        nowIndex = T == tau
        nowY, nowA = Y[nowIndex], A[nowIndex]
        diedC, diedT = np.sum(nowY[~nowA]),  np.sum(nowY[nowA])
        atRiskC, atRiskT = np.sum(T[~A] >= tau), np.sum(T[A] >= tau)
        diedC = diedC*atRiskT/atRiskC
        saved[i] = diedC - diedT
    
    score = np.sum(saved)
    return score

    
    