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
