import numpy as np
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')

a= np.ones((10,10,10))
a[:,:,5]=0
z,y,x = a.nonzero()
ax.scatter3D(x, y, z, c=z, cmap='Greens');
plt.show()