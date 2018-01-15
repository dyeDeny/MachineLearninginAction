# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 21:41:23 2018

@author: dye
"""

import numpy as np

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet= set([])
    for document in dataSet:
        vocabSet = vocabSet.union(set(document))
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returVec[vocabList.index(word)] = 1
        else:
            print("word \"%s\" is not in vocabList" % (word))
    return returVec

def bagOfWords2Vec(vocabList, inputSet):
    returVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returVec[vocabList.index(word)] += 1
        else:
            print("word \"%s\" is not in vocabList" % (word))
    return returVec

def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W+', bigString)
    return [token.lower() for token in listOfTokens if len(token) > 3]

def trainNB0(trainMatrix, trainCategory):
    #numTrainDocs = len(trainMatrix)
    #numWords = len(trainMatrix[0])
    numTrainDocs = trainMatrix.shape[0]
    numWords = trainMatrix.shape[1]
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = np.zeros(numWords)
    p1Num = np.zeros(numWords)
    p0Denom = 0.0
    p1Denom = 0.0
    for i in range(numTrainDocs):
        if 1 == trainCategory[i]:
            p1Num += trainMatrix[i]
            p1Denom += np.sum(trainMatrix[i, :])
        else:
            p0Num += trainMatrix[i]
            p0Denom += np.sum(trainMatrix[i, :])
    p1Vect = np.log(p1Num / p1Denom + 10**-10) #RuntimeWarning: divide by zero encountered in log
    p0Vect = np.log(p0Num / p0Denom + 10**-10)
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = np.sum(np.multiply(vec2Classify, p1Vec)) + np.log(pClass1)
    p0 = np.sum(np.multiply(vec2Classify, p0Vec)) + np.log(1- pClass1)
    
    return (p1 > p0 and 1 or 0)

def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open("email/spam/%d.txt" % (i), "rb").read().decode("UTF-8"))
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open("email/ham/%d.txt" % (i), "rb").read().decode("UTF-8"))
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = np.random.randint(len(trainingSet))
        testSet.append(trainingSet[randIndex])
        trainingSet.remove(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docId in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docId]))
        trainClasses.append(classList[docId])
    p0Vect, p1Vect, pAbusive =  trainNB0(np.array(trainMat), trainClasses)
    errCnt = 0
    for docId in testSet:
        wordVec = setOfWords2Vec(vocabList, docList[docId])
        if classifyNB(wordVec, p0Vect, p1Vect, pAbusive) != classList[docId]:
            errCnt += 1
    print("error rate is %f" % (errCnt / float(len(testSet))))

def test():
    postingList,classVec = loadDataSet()
    vocabList = createVocabList(postingList)
    print("vocabList:" + str(vocabList))
    vec = setOfWords2Vec(vocabList, postingList[0])
    print(vec)

def testNB():
    postingList,classVec = loadDataSet()
    vocabList = createVocabList(postingList)
    trainMat = []
    for postintDoc in postingList:
        trainMat.append(setOfWords2Vec(vocabList, postintDoc))
    p0Vect, p1Vect, pAbusive =  trainNB0(np.array(trainMat), classVec)
    testEntry1 = ['love', 'my', 'dalmation']
    testDoc1 = setOfWords2Vec(vocabList, testEntry1)
    pTest1 = classifyNB(testDoc1, p0Vect, p1Vect, pAbusive)
    print("pTest1 is classified as %d" % (pTest1))
    
    testEntry2 = ['stupid', 'garbage']
    testDoc2 = setOfWords2Vec(vocabList, testEntry2)
    pTest2 = classifyNB(testDoc2, p0Vect, p1Vect, pAbusive)
    print("pTest2 is classified as %d" % (pTest2))

if "__main__" == __name__:
    spamTest()
