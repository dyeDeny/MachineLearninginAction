# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 21:58:03 2018

@author: dye
"""

import numpy as np

def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open("testSet.txt")
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def gradAscent(dataIn, classLabels):
    dataMat = np.mat(dataIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = dataMat.shape
    alpha = 0.01
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        a = sigmoid(np.dot(dataMat, weights))
        error = (labelMat - a)  # loss = -(y*log(a) + (1-y)*log(1-a))
        weights = weights + alpha * dataMat.transpose() * error / m
    return weights

def stocGradAscent0(dataIn, classLabels):
    dataMat = np.mat(dataIn)
    m, n = dataMat.shape
    alpha = 0.01
    weights = np.ones((n, 1))
    for i in range(m):
        a = sigmoid(np.dot(dataMat[i, :], weights))
        error = (classLabels[i] - a)  # loss = -(y*log(a) + (1-y)*log(1-a))
        weights = weights + alpha * \
                        np.multiply(error, np.transpose(dataMat[i, :]))
    return weights

def stocGradAscent1(dataIn, classLabels, numIter = 150):
    dataMat = np.mat(dataIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = dataMat.shape
    alpha = 0.01
    weights = np.ones((n, 1))
    for j in range(numIter):
        permutation = np.random.permutation(m)
        dataMat = dataMat[permutation, :]
        labelMat = labelMat[permutation, :]
        for i in range(m):
            alpha = 4.0 / (i + j + 1.0) + 0.01
            a = sigmoid(np.dot(dataMat[i, :], weights))
            error = (labelMat[i, :] - a)  # loss = -(y*log(a) + (1-y)*log(1-a))
            weights = weights + alpha * \
                            np.multiply(error, np.transpose(dataMat[i, :]))
    return weights

def classifyVector(x, weights):
    prob = np.squeeze(sigmoid(np.dot(x, weights)))
    if (prob > 0.5):
        return 1.0
    else:
        return 0.0

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataIn, labelVec = loadDataSet()
    dataArr = np.array(dataIn)
    m = dataArr.shape[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(m):
        if (labelVec[i] == 1):
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(xcord1, ycord1, s=30, c="red", marker="s")
    ax1.scatter(xcord2, ycord2, s=30, c="green", marker="v")
    x = np.arange(-3.0, 3.0, 0.1)
    y = -(weights[0] + weights[1] * x) / weights[2]
    ax1.plot(x, y.transpose())
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def colicTest():
    frTrain = open("horseColicTraining.txt")
    frTest = open("horseColicTraining.txt")
    trainSet = []
    trainLabels = []
    for line in frTrain.readlines():
        lineVec = line.strip().split('\t')
        lineFloat = list(map(float, lineVec))
        trainSet.append(lineFloat[:-1])
        trainLabels.append(lineFloat[-1])

    trainWeights = stocGradAscent1(trainSet, trainLabels, 500)
    errCnt = 0
    numTestVec = 0.0
    
    for line in frTest.readlines():
        numTestVec += 1
        lineVec = line.strip().split('\t')
        lineFloat = list(map(float, lineVec))
        if int(classifyVector(lineFloat[:-1], trainWeights)) \
            != int(lineFloat[-1]):
            errCnt += 1
    errRate = errCnt / numTestVec
    print("errRate is %f" % (errRate))
    return errRate

def test():
    dataArr, labelVec = loadDataSet()
    weights = stocGradAscent1(dataArr, labelVec)
    plotBestFit(weights)
    print(classifyVector([1, 1, 10], weights))

if "__main__" == __name__:
    test()
    colicTest()