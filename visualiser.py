from numpy import load
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
cv2.namedWindow('Optical',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Optical', 200,200)
cv2.namedWindow('Radar',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Radar', 200,200)
cv2.namedWindow('Opticaledge',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Opticaledge', 200,200)
# Window name in which image is displayed
window_name = 'Image'
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (255, 0, 0)
org = (50, 50)
org1 = (90, 50)
thickness = 2

OpticalData=load("motions\Optical.npy")
RadarData=np.transpose(load("motions\Radar.npy"), (0,1,3,2,4))
label=load("motions\Labels.npy")


print (OpticalData.shape)

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


while True:
    for motion in range(0,OpticalData.shape[0]):
        for i,b in zip(RadarData[motion,:,:,:],OpticalData[motion,:,:,:]):
            gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            auto = auto_canny(blurred)


            cv2.imshow('Optical',b)
            cv2.imshow('Opticaledge', auto)
            cv2.imshow('Radar', i)
            time.sleep(0.03)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    # plt.imshow(i,cmap="ocean", vmin=0)
    # plt.show()
# print (RadarData.shape)
# x,y,z= RadarData[1,:,:,:,1].nonzero()
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# ax.scatter(z,x , y, c= 'blue')
# plt.show()
# for alpha,b in zip(range(0,40),label):
#
#     for i in array[alpha,:,:,:]:
#         plt.imshow(i,cmap="ocean", vmin=0)
#         plt.ylabel(b)
#         plt.draw()
#         plt.pause(0.03)
#         plt.clf()
