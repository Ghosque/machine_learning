import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


def make_data(N, f=0.3, rseed=1):
    rand = np.random.RandomState(rseed)
    x = rand.randn(N)
    x[int(f * N)] += 5
    return x

x = make_data(1000)

hist = plt.hist(x, bins=30, normed=True)
plt.show()

density, bins, patches = hist
widths = bins[1:] - bins[:-1]
print((density * widths).sum())

x = make_data(20)
bins = np.linspace(-5, 10, 10)

fig, ax =plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True,
                      subplot_kw={'xlim': (-4, 9), 'ylim': (-0.02, 0.3)})
fig.subplots_adjust(wspace=0.05)
for i, offset in enumerate([0.0, 0.6]):
    ax[i].hist(x, bins=bins + offset, normed=True)
    ax[i].plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
plt.show()








