import numpy as np
from matplotlib import pyplot as plt

dat = np.loadtxt("dat_grid")
#print(dat)
#print(dat.shape)
#print(dat[:,0])

x = np.linspace(0, 1, num=dat.shape[0])
y = np.linspace(0, 1, num=dat.shape[1])
X, Y = np.meshgrid(x, y)
#print(X.shape)

plt.contour(X, Y, dat, levels=0, colors='k')
#plt.show()

#

dat = np.loadtxt("dat_point")
#print(dat.shape)
#print(dat)
#print(dat[:,0])

col = []
for y in dat[:, 2]:
    if y > 0:
        col.append('r')
    else:
        col.append('b')

sz = []
for a in dat[:, 3]:
    if a > 0.001:
        sz.append(50)
    else:
        sz.append(5)

plt.scatter(dat[:, 0], dat[:, 1], s=sz, c=col)

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.show()
