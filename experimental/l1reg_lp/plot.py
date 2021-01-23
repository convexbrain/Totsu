import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.gca(projection='3d')

dat = np.loadtxt("dat_grid")
#print(dat)
#print(dat.shape)
#print(dat[:,0])

x = np.linspace(0, 1, num=dat.shape[0])
y = np.linspace(0, 1, num=dat.shape[1])
X, Y = np.meshgrid(x, y)
#print(X.shape)

ax.plot_wireframe(X, Y, dat, linewidth=1, colors='g')
#plt.show()

#

dat = np.loadtxt("dat_point")
#print(dat.shape)
#print(dat)
#print(dat[:,0])

sz = []
for a in dat[:, 3]:
    if abs(a) > 0.001:
        sz.append(100)
    else:
        sz.append(10)

ax.scatter(dat[:, 0], dat[:, 1], dat[:, 2], s=sz, c='r')

plt.show()
