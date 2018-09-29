from scipy import interpolate
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


path = r'E:\Python\Ghosque\data\scipy_test.png'
im = np.array(Image.open(path).convert('L'), 'f')
print(im)
print(im.shape)
plt.imshow(im)
plt.show()
x = np.linspace(0, 1, im.shape[1])
z = interpolate.interp1d(x, im)
print(z)
print(type(z))




