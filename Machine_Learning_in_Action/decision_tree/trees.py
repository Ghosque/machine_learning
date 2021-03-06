import math
import operator
import matplotlib.pyplot as plt
import treePlotter as tp


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def calcShannonEnt(dataSet):
    """
    计算香农熵
    """
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]/numEntries)
        shannonEnt -= prob * math.log(prob, 2)

    return shannonEnt


def splitDataSet(dataSet, axis, value):
    """
    划分数据集
    :param dataSet: 数据集
    :param axis: 划分特征所在列
    :param value: 需要返回的特征的值
    """
    retDataSet = []
    for featVec in dataSet:
        # 抽出符合条件的值
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)

    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    """
    选择最好的数据集划分方式
    """
    numFeatures = len(dataSet[0]) - 1
    # 原始香农熵
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i

    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)

    return myTree


def classify(inputTree, featLabels, testVec):
    """
    测试算法
    :param inputTree: 输入树
    :param featLabels: 标签列表
    :param testVec: 测试向量
    :return: result
    """
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]

    return classLabel


def storeTreee(inputTree, filename):
    """存储树"""
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(inputTree, f)


def grabTree(filename):
    """读取树"""
    import pickle
    with open(filename, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    # myDat, labels = createDataSet()
    # print(labels)
    myTree = tp.retrieveTree(0)
    print(myTree)
    # result = classify(myTree, labels, [1, 1])
    # print(result)

    # storeTreee(myTree, 'classifierStorage.txt')
    tree = grabTree('classifierStorage.txt')
    print(tree)


