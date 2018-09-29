import numpy as np
import matplotlib.pyplot as plt

n_dots = 40
X = 5 * np.random.rand(n_dots, 1)
y = np.cos(X).ravel()
# 添加一些噪声
y += 0.2 * np.random.rand(n_dots) - 0.1

# 训练模型
from sklearn.neighbors import KNeighborsRegressor
k = 5
knn = KNeighborsRegressor(k)
knn.fit(X, y)

# 生成足够密集的点并进行预测
T = np.linspace(0, 5, 500)[:, np.newaxis]
y_pred = knn.predict(T)
score = knn.score(X, y)
print(score)

# 画出拟合曲线
plt.figure(figsize=(16, 10), dpi=144)
plt.scatter(X, y, c='g', label='data', s=100)  # 画出训练样本
plt.plot(T, y_pred, c='k', label='prediction', lw=4)  # 画出拟合曲线
plt.axis('tight')
plt.title('KNeighborsRegressor (k = {})'.format(k))
plt.show()










