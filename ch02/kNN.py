# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 21:38:12 2018

@author: dye
"""

import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    lables = ['A', 'A', 'B', 'B']
    return group, lables

def classify0(inX, dataSet, lables, k):
    dataSetSize = dataSet.shape[0]
    posX = np.tile(inX, (dataSetSize, 1)) - dataSet
    #print("posX:" + str(posX))
    dist = np.sqrt(np.sum(np.square(posX), axis=1))
    #print("dist:" + str(dist))
    sortedDistIndicies = np.argsort(dist)
    #print("sortedDistIndicies:" + str(sortedDistIndicies))
    classCount = {}
    for i in range(k):
        clz = lables[sortedDistIndicies[i]]
        classCount[clz] = classCount.get(clz, 0) + 1
    #print("classCount:" + str(classCount))
    return max(classCount.items(), key=lambda x:x[1])[0]

def file2mat(filename):
    fr = open(filename)
    arrayLines = fr.readlines()
    numOfLines = len(arrayLines)
    outMat = np.zeros((numOfLines, 3))
    outLabels = []
    for i in range(numOfLines):
        line = arrayLines[i]
        data = line.strip().split("\t")
        outMat[i, :] = data[0:-1]
        outLabels.append(int(data[-1]))
    return outMat, outLabels

def autoNorm(dataMat):
    minVal = np.min(dataMat, axis=0)
    maxVal = np.max(dataMat, axis=0)
    ranges = maxVal - minVal
    
    normData = (dataMat - minVal) / ranges
    return normData, ranges, minVal

def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLables = file2mat("datingTestSet2.txt")
    normData, ranges, minVal = autoNorm(datingDataMat)
    m = normData.shape[0]
    numTest = int(m*hoRatio)
    errCnt = 0
    
    for i in range(numTest):
        predict = classify0(normData[i, :], normData[numTest:, :], datingLables[numTest:], 3)
        print("predict :%d, real answer is:%d" % (predict, datingLables[i]))
        if (predict != datingLables[i]):
            errCnt += 1
    print("Total error rate: %f" % (errCnt/float(numTest)))

def test():
    datingDataMat, datingLables = file2mat("datingTestSet2.txt")
    normData, ranges, minVal = autoNorm(datingDataMat)
    print("normData:" + str(normData))
    print("datingLables:" + str(datingLables[0:20]))
    fig1 = plt.figure(1)
    fig2 = plt.figure(2)
    ax1 = fig1.add_subplot(111)
    ax1.scatter(normData[:, 0], normData[:, 1], 15.0*np.array(datingLables), 15.0*np.array(datingLables))
    ax2 = fig2.add_subplot(111)
    ax2.scatter(normData[:, 1], normData[:, 2], 15.0*np.array(datingLables), 15.0*np.array(datingLables))
    
    plt.show()
    print("done")

if __name__ == "__main__":
    datingClassTest()