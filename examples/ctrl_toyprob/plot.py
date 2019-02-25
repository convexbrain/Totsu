import numpy as np
from matplotlib import pyplot as plt

dat = np.loadtxt("dat_point")
#print(dat.shape)
#print(dat)
#print(dat[:,0])

plt.plot(dat[:, 0], dat[:, 1], marker='o')

#plt.xlim(0, 1)
#plt.ylim(0, 1)
plt.show()

t = dat.shape[0]
vx = dat[1: t, 0] - dat[0: t - 1, 0]
vy = dat[1: t, 1] - dat[0: t - 1, 1]
v = np.sqrt(vx * vx + vy * vy)
ax = dat[2: t, 0] - 2 * dat[1: t - 1, 0] + dat[0: t - 2, 0]
ay = dat[2: t, 1] - 2 * dat[1: t - 1, 1] + dat[0: t - 2, 1]
a = np.sqrt(ax * ax + ay * ay)

plt.plot(v, marker='o')
plt.plot(a, marker='o')
plt.show()
