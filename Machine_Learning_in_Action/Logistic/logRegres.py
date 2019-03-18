import random
import numpy as np
import matplotlib.pyplot as plt


def loadDataSet():
    dataMat = []
    labelMat = []

    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))

    return dataMat, labelMat


def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        wTx = dataMatrix * weights  # 计算出每个样本的wTx矩阵
        wTarr = np.array(wTx)  # 转换为list  方便以下遍历
        actArr = [sigmoid(warr) for warr in wTarr]  # 迭代遍历然后存入列表
        actMat = np.mat([actArr]).reshape(m, 1)  # 把列表转化为矩阵后，然后把矩阵变为（）
        error = (labelMat - actMat)
        weights = weights + alpha * dataMatrix.transpose() * error

    return weights


def plotBestFit(wei):
    weights = wei.getA()
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def stocGradAscent0(dataMatrix, classLabels):
    """随机梯度上升算法"""
    dataMatrix = np.array(dataMatrix)  # 把处理后的列表转化为数组
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights+alpha * error * dataMatrix[i]
    # 这个目的是方便下面plotBestFit()函数中，wei.getA()的操作，其中的wei只能为矩阵
    weights = np.mat(weights).reshape((3, 1))

    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    """随机梯度上升算法"""
    dataMatrix = np.array(dataMatrix)  # 把处理后的列表转化为数组
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights+alpha * error * dataMatrix[randIndex]
            del dataIndex[randIndex]
    # 这个目的是方便下面plotBestFit()函数中，wei.getA()的操作，其中的wei只能为矩阵
    # weights = np.mat(weights).reshape((3, 1))

    return weights


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    return 0.0


def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currentLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currentLine[i]))

        trainingSet.append(lineArr)
        trainingLabels.append(float(currentLine[21]))

    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currentLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currentLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights)) != int(currentLine[21]):
            errorCount += 1

    errorRate = float(errorCount) / numTestVec
    print("The error rate of this test is %f" % errorRate)

    return errorRate


def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()

    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))


if __name__ == '__main__':
    multiTest()


