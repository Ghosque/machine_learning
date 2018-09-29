
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

# k近邻分类器
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=1)

from sklearn.cross_validation import train_test_split
# 每个数据集分一半数据
X1, X2, y1, y2 = train_test_split(X, y, random_state=0, train_size=0.5)

# 用模型你和训练数据
model.fit(X1, y1)

# 在测试集中评估模型准确率
from sklearn.metrics import accuracy_score
y2_model = model.predict(X2)
score = accuracy_score(y2, y2_model)
print(score)

# 两轮交叉检验
y2_model = model.fit(X1, y1).predict(X2)
y1_model = model.fit(X2, y2).predict(X1)
print(accuracy_score(y1, y1_model), accuracy_score(y2, y2_model))

# 多轮交叉检验
from sklearn.cross_validation import cross_val_score
scores = cross_val_score(model, X, y, cv=5)
print(scores)

# LOO(leave-one-out)
from sklearn.cross_validation import LeaveOneOut
scores = cross_val_score(model, X, y, cv=LeaveOneOut(len(X)))
print(scores)
print(scores.mean())





