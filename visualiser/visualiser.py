from numpy import load
import numpy as np
import matplotlib.pyplot as plt
array=load("motions\Hand2.npy")
array2=load("motions\LSTMdata.npy")
label=load("motions\labelsHand2.npy")

print (array[10,:,:,:,1].shape)
# x,y = array2[46,:,:].nonzero()
x,y,z= array[12,:,:,:,1].nonzero()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(z,x , y, c= 'blue')
plt.show()
# for alpha,b in zip(range(0,40),label):
#
#     for i in array[alpha,:,:,:]:
#         plt.imshow(i,cmap="ocean", vmin=0)
#         plt.ylabel(b)
#         plt.draw()
#         plt.pause(0.03)
#         plt.clf()
