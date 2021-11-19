from numpy import load
import numpy as np
import matplotlib.pyplot as plt
OpticalData=load("motions\Optical.npy")
RadarData=load("motions\Radar.npy")
label=load("motions\Label.npy")

print (RadarData[10,:,:,:,1].shape)
x,y,z= RadarData[12,:,:,:,1].nonzero()
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
