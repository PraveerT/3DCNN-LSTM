from numpy import load
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import sys

OpticalData=load("motions\Optical.npy")
RadarData=np.transpose(load("motions\Radar.npy"), (0,1,3,2,4))
RadarData2=np.transpose(load("motions\Radar.npy"), (0,1,3,2,4))
label=load("motions\Labels.npy")


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged

print(OpticalData.shape)
arr=OpticalData.shape[0]*OpticalData.shape[1]
OpticalData.resize(arr,100,100,3)
output=np.empty((arr,100,100,3),dtype=np.uint8)


for frame,len in zip(OpticalData,range(len(OpticalData))):
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    auto = auto_canny(blurred)
    _,thresh1=cv2.threshold(blurred,127,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    data=np.stack([gray,blurred,thresh1],axis=2)
    output[len,:,:,:]=data


output.resize(RadarData.shape[0],RadarData.shape[1],100,100,3)
print (output.shape)
np.save('motions\ThreshGrayOpticalData.npy',output)
#
