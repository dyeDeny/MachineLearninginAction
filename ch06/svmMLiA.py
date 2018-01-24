# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 23:46:43 2018

@author: dye
"""
import numpy as np                                                                                                                                        

def loadDataSet(fileName):
    dataArr = []
    labelArr = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split("\t")
        dataArr.append([float(lineArr[0]), float(lineArr[1])])
        labelArr.append(float(lineArr[2]))
    return dataArr, labelArr

def selectJrand(i, m):
    j = i
    while(j == i):
        j = int(np.random.uniform(0,m))
    return j

def clipAlpha(a, L, H):
    if a > H:
        a = H
    if a < L:
        a = L
    return a

def smoSimple(dataArrIn, classLabels, C, toler, maxIter):
    dataMat = np.mat(dataArrIn)
    labelMat = np.mat(classLabels).transpose()
    b = 0.0
    m, n = dataMat.shape
    alphas = np.mat(np.zeros((m, 1)))
    iter = 0
    while (iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(np.multiply(alphas, labelMat).T * \
                        (dataMat * dataMat[i, :].T)) + b
            Ei = fXi - float(labelMat[i])
            #print(Ei)
            if (((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or \
                ((labelMat[i] * Ei > toler) and (alphas[i] > 0))):
                j = selectJrand(i, m) # Need to be optimized
                fXj = float(np.multiply(alphas, labelMat).T * \
                            (dataMat * dataMat[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, alphas[j] - alphas[i] + C)
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L == H")
                    continue
                #eta = K11 + K22 - 2*K12
                eta = dataMat[i, :] * dataMat[i, :].T + \
                        dataMat[j, :] * dataMat[j, :].T - \
                        2.0 * dataMat[i, :] * dataMat[j, :].T
                if eta <= 0:
                    print("eta <= 0")
                    continue
                alphaJunc = alphas[j] + labelMat[j] * (Ei -Ej) / eta
                alphas[j] = clipAlpha(alphaJunc, L, H)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("%d not moving enough" % (j))
                    continue
                # alphas[j]*labelMat[j] + alphas[i]*labelMat[i] = cosnt
                # and labels**2 = 1
                alphas[i] -= (alphas[j] - alphaJold) * labelMat[i] * labelMat[j]
                bi = b - Ei - \
                    labelMat[i] * (alphas[i] - alphaIold) * (dataMat[i, :] * dataMat[i, :].T) -\
                    labelMat[j] * (alphas[j] - alphaJold) * (dataMat[j, :] * dataMat[i, :].T)
                bj = b - Ej - \
                    labelMat[i] * (alphas[i] - alphaIold) * (dataMat[i, :] * dataMat[j, :].T) -\
                    labelMat[j] * (alphas[j] - alphaJold) * (dataMat[j, :] * dataMat[j, :].T)
                if (0 < alphas[i] and C > alphas[i]):
                    b = bi
                elif (0 < alphas[j] and C > alphas[j]):
                    b = bj
                else:
                    b = (bi + bj) / 2.0
                alphaPairsChanged += 1
                print("iter: %d, i: %d, paires changed: %d" % (iter, i, alphaPairsChanged))
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" % (iter))
    return b, alphas

def calcWs(alphas, dataArr, classLabels):
    X = np.mat(dataArr)
    labelMat = np.mat(classLabels).transpose()
    m, n = X.shape
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i]*labelMat[i], X[i, :].T)
        pass
    print("wshape: " + str(w.shape))
    return w

class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = dataMatIn.shape[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m, 2)))
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)

def calcEk(oS, k):
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * \
                (oS.K[:, k])) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJ(i, oS, Ei):
    maxK = -1
    maxDeltaE = 0.0
    Ej = 0.0
    oS.eCache[i] = [1, Ei]
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEcacheList) > 1):
        for k in validEcacheList:
            if (k == i):
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ek - Ei)
            if (deltaE > maxDeltaE):
                maxDeltaE = deltaE
                maxK = k
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
        return j, Ej

def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

def innerL(i, oS):
    Ei = calcEk(oS, i)
    if (((oS.labelMat[i] * Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or \
                ((oS.labelMat[i] * Ei > oS.tol) and (oS.alphas[i] > 0))):
        j, Ej = selectJ(i, oS, Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.alphas[j] - oS.alphas[i] + oS.C)
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print("L == H")
            return 0
        #eta = K11 + K22 - 2*K12
        eta = oS.K[i, i] + oS.K[j, j] - 2 * oS.K[i, j]
        if eta <= 0:
            print("eta <= 0")
            return 0
        alphaJunc = oS.alphas[j] + oS.labelMat[j] * (Ei -Ej) / eta
        oS.alphas[j] = clipAlpha(alphaJunc, L, H)
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):
            print("%d not moving enough" % (j))
            return 0
        # alphas[j]*labelMat[j] + alphas[i]*labelMat[i] = cosnt
        # and labels**2 = 1
        oS.alphas[i] -= (oS.alphas[j] - alphaJold) * oS.labelMat[i] * oS.labelMat[j]
        bi = oS.b - Ei - \
            oS.labelMat[i] * (oS.alphas[i] - alphaIold) * (oS.K[i, i]) -\
            oS.labelMat[j] * (oS.alphas[j] - alphaJold) * (oS.K[i, j])
        bj = oS.b - Ej - \
            oS.labelMat[i] * (oS.alphas[i] - alphaIold) * (oS.K[i, j]) -\
            oS.labelMat[j] * (oS.alphas[j] - alphaJold) * (oS.K[j, j])
        if (0 < oS.alphas[i] and oS.C > oS.alphas[i]):
            oS.b = bi
        elif (0 < oS.alphas[j] and oS.C > oS.alphas[j]):
            oS.b = bj
        else:
            oS.b = (bi + bj) / 2.0
        return 1
    else:
        return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup=("lin", 0)):
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if (entireSet):
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
            iter += 1
        else:
            nonBoundsIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundsIs:
                alphaPairsChanged += innerL(i, oS)
            iter += 1
        if (entireSet):
            entireSet = False
        elif (alphaPairsChanged == 0):
            entireSet = True
    return oS.b, oS.alphas

def kernelTrans(X, A, kTup):
    m, n = X.shape
    K = np.mat(np.zeros((m, 1)))
    if ("lin" == kTup[0]):
        K = X * A.T
    elif ("rbf" == kTup[0]):
        #delta = X - A
        #K = np.exp(-delta * delta.T / (float(kTup[1]) ** 2))[0,:]
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = np.exp(K / (-1*kTup[1] ** 2))
    else:
        raise Exception("Unrecognized kernel function")
    return K

def testRbf(k1=1.3):
    dataArr, labelArr = loadDataSet("testSetRBF.txt")
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ("rbf", k1))
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).transpose()
    svInd = np.nonzero(alphas.A>0)[0]
    supVecs = dataMat[svInd, :]
    labelSV = labelMat[svInd]
    m, n = dataMat.shape
    errCnt = 0
    for i in range(m):
        kernelEval = kernelTrans(supVecs, dataMat[i, :], ("rbf", k1))
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if (np.sign(predict) != np.sign(labelMat[i])):
            print("train[%d] predict is %f, label is %d" % (i, predict, labelMat[i]))
            errCnt += 1
    print("traning error is %f" % (float(errCnt) / m))
    
    testDataArr, testLabelArr = loadDataSet("testSetRBF2.txt")
    testDataMat = np.mat(dataArr)
    testLabelMat = np.mat(labelArr).transpose()
    mt = testDataMat.shape[0]
    errCnt = 0
    for i in range(mt):
        kernelEval = kernelTrans(supVecs, testDataMat[i, :], ("rbf", k1))
        predict = kernelEval.T * np.multiply(labelSV, alphas[svInd]) + b
        if (np.sign(predict) != np.sign(testLabelMat[i])):
            print("train[%d] predict is %f, label is %d" % (i, predict, testLabelMat[i]))
            errCnt += 1
    print("test error is %d/%d = %f" % (errCnt, mt, float(errCnt) / mt))

def test():
    dataArr, labelArr = loadDataSet("testSet.txt")
    b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)
    ws = calcWs(alphas, dataArr, labelArr)
    print("[calcWs]ws is %s" % (str(ws)))
    for i in range(100):
        if (alphas[i] > 0.0):
            print(dataArr[i], labelArr[i])
    return b, ws

if "__main__" == __name__:
    testRbf()