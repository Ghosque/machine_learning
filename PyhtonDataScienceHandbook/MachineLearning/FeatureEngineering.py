
"""分类特征"""
data = [
    {'price': 850000, 'rooms': 4, 'neighborhood': 'Queen Anne'},
    {'price': 700000, 'rooms': 3, 'neighborhood': 'Fremont'},
    {'price': 650000, 'rooms': 3, 'neighborhood': 'Wallingford'},
    {'price': 600000, 'rooms': 2, 'neighborhood': 'Fremont'}
]
# 独热编码
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False, dtype=int)
print(vec.fit_transform(data))
print(vec.get_feature_names())

vec = DictVectorizer(sparse=True, dtype=int)
print(vec.fit_transform(data))


"""文本特征"""
sample = ['problem of evil',
          'evil queen',
          'horizon problem']
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
X = vec.fit_transform(sample)
# 用带列标签的DataFrame表示
import pandas as pd
df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
print(df)
# TF-IDF term frequency-inverse document frequency 词频逆文档频率
from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer()
X = vec.fit_transform(sample)
df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
print(df)


"""衍生特征"""
import numpy as np
import matplotlib.pyplot as plt
x = np.array([1, 2, 3, 4, 5])
y = np.array([4, 2, 1, 3, 7])
plt.scatter(x, y)
# plt.show()
# 拟合
from sklearn.linear_model import LinearRegression
X = x[:, np.newaxis]
model = LinearRegression().fit(X, y)
yfit = model.predict(X)
plt.scatter(x, y)
plt.plot(x, yfit)
# plt.show()
# 多项式拟合
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=3, include_bias=False)
X2 = poly.fit_transform(X)
print(X2)
model = LinearRegression().fit(X2, y)
yfit = model.predict(X2)
plt.scatter(x, y)
plt.plot(x, yfit)
# plt.show()


"""缺失值填充"""
from numpy import nan
X = np.array([[nan, 0, 3],
              [3, 7, 9],
              [4, 5, 2],
              [4, nan, 6],
              [8, 8, 1]])
y = np.array([14, 16, -1, 8, -5])
from sklearn.preprocessing import Imputer
# 列均值替换缺失值
imp = Imputer(strategy='mean')
X2 = imp.fit_transform(X)
print(X2)
model = LinearRegression().fit(X2, y)
array = model.predict(X2)
print(array)


"""特征管道"""
from sklearn.pipeline import make_pipeline
model = make_pipeline(Imputer(strategy='mean'),
                      PolynomialFeatures(degree=2),
                      LinearRegression())
model.fit(X, y)
print(y)
print(model.predict(X))





