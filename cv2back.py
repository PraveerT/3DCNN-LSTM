from numpy import load
import cv2, time
import numpy as np

cv2.namedWindow('Optical',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Optical', 200,200)
backSub = cv2.createBackgroundSubtractorMOG2()
OpticalData=load('C:/Users/prav/Dropbox/source/Optical.npy')
print (OpticalData.shape)
OpticalData.resize(6000,100,100,3)
Output=np.empty((6000,100,100),dtype=np.uint8)
for frame,frameout in zip(OpticalData,range(0,len(Output))):
    fgMask = backSub.apply(frame)
    cv2.imshow('Optical',fgMask)
    Output[frameout]=fgMask
    time.sleep(0.03)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
Output.resize(100,60,100,100)


