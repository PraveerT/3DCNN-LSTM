from numpy import load
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import sys

cv2.namedWindow('Optical',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Optical', 200,200)
cv2.namedWindow('Gray',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Gray', 200,200)
cv2.namedWindow('Blurred',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Blurred', 200,200)
cv2.namedWindow('Thresh',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Thresh', 200,200)
cv2.namedWindow('Radar',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Radar', 200,200)
# Window name in which image is displayed
window_name = 'Image'
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (255, 0, 0)
org = (50, 50)
org1 = (90, 50)
thickness = 2
folder="C:/Users/prav/Dropbox/source"
OpticalData=load(folder+"/FgMask.npy")
RadarData=np.transpose(load(folder+"/Radar.npy"), (0,1,3,2,4))
label=load(folder+"/Labels.npy")
output=load(folder+"/ThreshGrayOpticalData.npy")
print (label)

print (OpticalData.shape,RadarData.shape)
print (OpticalData)
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged

while True:
    for motion in range(20,OpticalData.shape[0]):
        for frame in OpticalData[motion]:
            cv2.imshow('Optical',frame)

            time.sleep(0.03)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
