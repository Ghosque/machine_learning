import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('diabetes.csv')
print('dataset shape {}'.format(data.shape))
print(data.head())
print(data.groupby('Outcome').size())  # 阳性和阴性样本的个数

X = data.iloc[:, 0:8]
Y = data.iloc[:, 8]
print('shape of X: {}; shape of Y: {}'.format(X.shape, Y.shape))
# 划分训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# 模型比较
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
# 构造三个模型
models = []
models.append(('KNN', KNeighborsClassifier(n_neighbors=2)))
models.append(('KNN with weights', KNeighborsClassifier(n_neighbors=2, weights='distance')))
models.append(('Radius Neighbors', RadiusNeighborsClassifier(n_neighbors=2, radius=500.0)))

# 分别训练3个模型，并计算评分
results = []
for name, model in models:
    model.fit(X_train, Y_train)
    results.append((name, model.score(X_test, Y_test)))
for i in range(len(results)):
    print('name: {}; score: {}'.format(results[i][0], results[i][1]))

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

results = []
for name, model in models:
    kfold = KFold(n_splits=10)
    cv_result = cross_val_score(model, X, Y, cv=kfold)
    results.append((name, cv_result))
for i in range(len(results)):
    print('name: {}; cross val score: {}'.format(results[i][0], results[i][1].mean()))

# 结果显示普通的k-均值算法性能更优，即第一种

# 模型训练
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, Y_train)
train_score = knn.score(X_train, Y_train)
test_score = knn.score(X_test, Y_test)
print('train score: {}; test score: {}'.format(train_score, test_score))
# 评分低：1、拟合情况不佳，模型太简单，2、模型准确性欠佳

# 特征选择
from sklearn.feature_selection import SelectKBest

selector = SelectKBest(k=2)
X_new = selector.fit_transform(X, Y)
print(X_new[0: 5])

results = []
for name, model in models:
    kfold = KFold(n_splits=10)
    cv_result = cross_val_score(model, X_new, Y, cv=kfold)
    results.append((name, cv_result))
for i in range(len(results)):
    print('name: {}; cross val score: {}'.format(results[i][0], results[i][1].mean()))
# 评分结果是普通的k-均值算法准确性高

# 画出数据
plt.figure(figsize=(10, 6), dpi=200)
plt.ylabel('BMI')
plt.xlabel('Glucose')
# 画出Y == 0的阴性样本，用圆圈表示
plt.scatter(X_new[Y==0][:, 0], X_new[Y==0][:, 1], c='r', s=20, marker='o')
# 画出Y == 1的阳性样本，用三角形表示
plt.scatter(X_new[Y==1][:, 0], X_new[Y==1][:, 1], c='g', s=20, marker='^')
plt.show()








