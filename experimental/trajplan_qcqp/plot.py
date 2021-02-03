import numpy as np
from matplotlib import pyplot as plt

dat = np.loadtxt("dat_point")
#print(dat.shape)
#print(dat)
#print(dat[:,0])
dat_t = np.loadtxt("dat_target")

if True:
    plt.plot(dat_t[:, 0], dat_t[:, 1], 'r*', markersize=20, markerfacecolor='w')
    plt.plot(dat[:, 1], dat[:, 2], marker='o')
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.show()

#####

dt = dat[1, 0] - dat[0, 0]
t = dat.shape[0]

vx = dat[1: t, 1] - dat[0: t - 1, 1]
vy = dat[1: t, 2] - dat[0: t - 1, 2]
v = np.sqrt(vx * vx + vy * vy) / dt

ax = dat[2: t, 1] - 2 * dat[1: t - 1, 1] + dat[0: t - 2, 1]
ay = dat[2: t, 2] - 2 * dat[1: t - 1, 2] + dat[0: t - 2, 2]
a = np.sqrt(ax * ax + ay * ay) / dt / dt

if True:
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(dat[0: t - 1, 0], v, marker='.')
    ax[1].plot(dat[0: t - 2, 0], a, marker='.')
    ax[1].set_xlim(0, 1)
    ax[0].set_xlim(0, 1)
    ax[1].set_xlabel('time')
    ax[0].set_ylabel('velocity mag.')
    ax[1].set_ylabel('acceleration mag.')
    plt.show()
