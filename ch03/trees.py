# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 20:15:26 2018

@author: dye
"""

from math import log
import operator as op

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    featureDict = {}
    shannonEnt = 0
    for i in range(numEntries):
        label = dataSet[i][-1]
        if label not in featureDict.keys():
            featureDict[label] = 0
        featureDict[label] += 1
    for key in featureDict.keys():
        prob = featureDict[key] / numEntries
        shannonEnt += -prob * log(prob, 2)
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if (featVec[axis] == value):
            reducedVec = featVec[:axis]
            reducedVec.extend(featVec[axis+1:])
            retDataSet.append(reducedVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeat = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestInfoFeat = -1
    for i in range(numFeat):
        uniqFeats = set([featVec[i] for featVec in dataSet])
        newEnt = 0.0
        for feat in uniqFeats:
            retDataSet = splitDataSet(dataSet, i, feat)
            prop = len(retDataSet) / float(len(dataSet))
            newEnt += prop * calcShannonEnt(retDataSet)
        infoGain = baseEntropy - newEnt
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestInfoFeat = i
    return bestInfoFeat

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=lambda x:x[1], reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    classList = [featVec[-1] for featVec in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestLabel = labels[bestFeat]
    myTree = {bestLabel:{}}
    labels.remove(bestLabel)
    #del(labels[bestFeat])
    featVals = set([featVec[bestFeat] for featVec in dataSet])
    for featVal in featVals:
        subLabels = labels[:]
        myTree[bestLabel][featVal] = createTree(\
              splitDataSet(dataSet, bestFeat, featVal), subLabels)
    return myTree

def createDataSet():
    dataSet = [[1, 1, "yes"],
               [1, 1, "yes"],
               [1, 0, "no"],
               [0, 1, "no"],
               [0, 1, "no"]]
    labels = ["no surfacing", "flippers"]
    return dataSet, labels

def classify(inputTree, featLabels, testVec):
    rootNode = list(inputTree.keys())[0]
    secondDict = inputTree[rootNode]
    featIndex = featLabels.index(rootNode)
    
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if isinstance(secondDict[key], dict):
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, "wb")
    pickle.dump(inputTree, fw)
    fw.close

def grabTree(filename):
    import pickle
    fr = open(filename, "rb")
    return pickle.load(fr)

def test():
    data, labels= createDataSet()
    shannonEnt = calcShannonEnt(data)
    print("shannonEnt is:" + str(shannonEnt))
    '''
    feat = chooseBestFeatureToSplit(data)
    print("Best feature:%d" % (feat))
    mlabel = majorityCnt(labels)
    print("mlabel is:%s" % (mlabel))
    '''
    print("data:%s, labels:%s" % (str(data), str(labels)))
    myTree = createTree(data, labels)
    print("myTree is:%s" % (myTree))

if "__main__" == __name__:
    test()
        