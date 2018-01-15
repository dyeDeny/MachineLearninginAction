# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 22:07:53 2018

@author: dye
"""

import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords="axes fraction",\
                           xytext=centerPt, textcoords="axes fraction",\
                           va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)
'''
def createPlot():
    fig = plt.figure(1, facecolor="white")
    fig.clf()
    createPlot.ax1 = fig.add_subplot(111, frameon=False)
    plotNode("Decision Node", (0.5, 0.1), (0.2, 0.2), decisionNode)
    plotNode("Leaf Node", (0.8, 0.1), (0.5, 0.8), leafNode)
    plt.show()
'''

def getNumLeafs(myTree):
    numLeafs = 0
    rootNode = list(myTree.keys())[0]
    secondDict = myTree[rootNode]
    for key in secondDict.keys():
        if isinstance(secondDict[key], dict):
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    depth = 0
    rootNode = list(myTree.keys())[0]
    secondDict = myTree[rootNode]
    for key in secondDict.keys():
        if isinstance(secondDict[key], dict):
            tmpDepth = 1 + getTreeDepth(secondDict[key])
        else:
            tmpDepth = 1
        #print("max(%d, %d) = %d" % (tmpDepth, depth, max(tmpDepth, depth)))
        depth = max(tmpDepth, depth)
    return depth

def retriveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'HEAD': {0: 'no', 1: 'yes'}}, 1: 'no'}}}},
                   {'no surfacing': {0: 'no', 1: 'yes'}}
            ]
    return listOfTrees[i]

def plotMidText(cntrPt, parentPt, string):
    xMid = (parentPt[0] + cntrPt[0]) / 2.0
    yMid = (parentPt[1] + cntrPt[1]) / 2.0
    createPlot.ax1.text(xMid, yMid, string)


def plotTree(myTree, parentPt, nodeTxt):
    leafs = getNumLeafs(myTree)
    rootNode = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(leafs)) / 2.0 / plotTree.totalW, \
              plotTree.yOff)
    
    print("myTree (%f, %f), parentPt (%f, %f), nodeTxt (%s)" % \
           (cntrPt[0], cntrPt[1], parentPt[0], parentPt[1], str(nodeTxt)))
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(rootNode, cntrPt, parentPt, decisionNode)
    
    secondDict = myTree[rootNode]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    
    for key in secondDict.keys():
        if isinstance(secondDict[key], dict):
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD

def createPlot(inTree):
    fig = plt.figure(1, facecolor="white")
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), "")
    plt.show()

def test():
    myTree = retriveTree(2)
    leafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    print("leafs:%d" % (leafs))
    print("depth:%d" % (depth))
    createPlot(myTree)
    
if __name__ == "__main__":
    test()