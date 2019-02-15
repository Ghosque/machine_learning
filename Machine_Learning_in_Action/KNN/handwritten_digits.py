import os
import operator
import numpy as np


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


def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    fr.close()
    return returnVect


def hanwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        classNumStr = int(fileNameStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/{}'.format(fileNameStr))
    testFileList = os.listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNumStr = int(fileNameStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/{}'.format(fileNameStr))
        classifierResult = classify(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d\n" %
              (classifierResult, classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1.0
    print("the total number of errors is: %d\n" % errorCount)
    print("the total error rate is: %f" % (errorCount/float(mTest)))


if __name__ == '__main__':
    hanwritingClassTest()


