import numpy as np
import matplotlib.pyplot as plt


def loadSimpData():
    datMat = np.mat([[1.0, 2.1],
                     [2.0, 1.1],
                     [1.3, 1.0],
                     [1.0, 1.0],
                     [2.0, 1.0]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]

    return datMat, classLabels


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0

    return retArray


def buildStimp(dataArr, classLabels, D):
    dataMatrix = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = np.mat(np.zeros((m, 1)))
    minError = np.inf
    # 遍历特征
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        # 遍历同特征的所有值
        for j in range(-1, int(numSteps)+1):
            # 切换不等式
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                print("split: dim %d, thresh %.2f, thresh inequal: %s, the weighted error is %.3f" %
                      (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal

    return bestStump, minError, bestClasEst


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    """
    基于单层决策树的AdaBoost训练过程
    :param dataArr: 数据集
    :param classLabels: 类别标签
    :param numIt: 迭代次数
    :return:
    """
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m, 1))/m)
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStimp(dataArr, classLabels, D)
        print("D:", D.T)
        alpha = float(0.5 * np.log((1.0-error)/max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print("classEst:", classEst.T)
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        aggClassEst += alpha * classEst
        print("aggClassEst:", aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print("total error:", errorRate, "\n")
        if errorRate == 0.0:
            break

    return weakClassArr, aggClassEst


def adaClassify(datToClass, classifierArr):
    dataMatrix = np.mat(datToClass)
    m = dataMatrix.shape[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,
                                 classifierArr[i]['dim'],
                                 classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'],)
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print(aggClassEst)

    return np.sign(aggClassEst)


def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))

    return dataMat, labelMat


def plotROC(predStrengths, classLbels):
    cur = (1.0, 1.0)
    ySum = 0.0
    numPosClas = sum(np.array(classLbels) == 1.0)
    yStep = 1 / float(numPosClas)
    xStep = 1 / float(len(classLbels) - numPosClas)
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLbels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0], cur[0]-delX], [cur[1], cur[1]-delY], c='b')
        cur = (cur[0] - delX, cur[1] - delY)

    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve of AdaBoost Horse Colic Detection System')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print("the Area Under the curve is:", ySum * xStep)


if __name__ == '__main__':
    # datMat, classLabels = loadSimpData()
    # D = np.mat(np.ones((5, 1))/5)

    # bestStump, minError, bestClasEst = buildStimp(datMat, classLabels, D)
    # print(bestStump, minError)
    # print(bestClasEst)

    # classifierArray = adaBoostTrainDS(datMat, classLabels, 9)
    # print(classifierArray)

    # adaClassify([[5, 0], [0, 0]], classifierArray)

    # datArr, labelArr = loadDataSet('horseColicTraining.txt')
    # classifierArray = adaBoostTrainDS(datArr, labelArr, 10)
    # testArr, testLabelArr = loadDataSet('horseColicTest.txt')
    # prediction10 = adaClassify(testArr, classifierArray)
    # errArr = np.mat(np.ones((67, 1)))
    # print(errArr[prediction10 != np.mat(testLabelArr).T].sum()/67)

    datArr, labelArr = loadDataSet('horseColicTraining.txt')
    classifierArray, aggClassEst = adaBoostTrainDS(datArr, labelArr, 10)
    plotROC(aggClassEst.T, labelArr)
