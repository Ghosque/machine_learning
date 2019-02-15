import operator
import matplotlib
import numpy as np
import matplotlib.pyplot as plt


def create_data_set():
    """
    创建已知数据
    :return: 已知数据集
    """
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']

    return group, labels


def classify(inX, dataSet, labels, k):
    """
    构造分类器
    :param inX: 输入向量
    :param dataSet: 训练样本集
    :param labels: 标签向量
    :param k: 选择最近邻居数目
    :return: 输出分类结果
    """
    dataSetSize = dataSet.shape[0]
    # 展开至shape=(dataSetSize, 1)，然后相减后平方相加再开方
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    # 排序后将对应位置保存在list中
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # key=operator.itemgetter(1)表示按第二个域进行排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]


def file2matrix(filename):
    """
    解析文件
    """
    fr = open(filename)
    # 获取行信息
    arrayOLines = fr.readlines()
    # 获取行数
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    # 行信息保存为numpy.array
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    """
    归一化特征值
    """
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    # newValues = (oldValue-min)/(max-min)
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    """
    算法测试
    """
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify(normMat[i, :], normMat[numTestVecs:m, :],
                                    datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" %
              (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))


def classifyPerson():
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')

    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))

    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print("You will probably like this person:", resultList[classifierResult-1])


if __name__ == '__main__':
    classifyPerson()


