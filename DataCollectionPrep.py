#motion #frame #val
import numpy as np
from numpy import save,load
import time
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import pickle as Pickle
import random
import csv
from radar import read
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()

def Average(lst):
    return sum(lst) / len(lst)

#0 youtube
#1 scroll down
#2 scroll up
#3 nothing
#4 pause
pred=0
inputsamples=1
inputname=5

motionname='Hand2'
Frames=60
label=np.array([inputname]*inputsamples)
def run():
    motion,name,opticalmotion=read.perform(inputname,inputsamples,Frames)

    xmap = interp1d([-0.2,0.2],[0,99],bounds_error=False,fill_value=(0,99),kind='linear')
    ymap = interp1d([0,0.6],[0,99],bounds_error=False,fill_value=(0,99),kind='linear')

    RadarMotionArray=[]
    OpticalMotionArray=[]
    matrixav=np.zeros((inputsamples,Frames,5))
    for alpha in range(1,inputsamples+1):

        RadarFrameArray=[]
        OpticalFrameArray=[]
        for frame in range(0,Frames):
            matrix = np.zeros((100, 100,3),dtype=np.uint8)
            xpositions=motion[str(alpha)][frame]['x']
            ypositions=motion[str(alpha)][frame]['y']
            dopplers=motion[str(alpha)][frame]['doppler']
            valrange = motion[str(alpha)][frame]['range']
            peakVal = motion[str(alpha)][frame]['peakVal']
            AvPosx=Average(xpositions)
            AvPosy=Average(ypositions)
            AvDoppler=Average(dopplers)
            AvValrange=Average(valrange)
            AvPeakVal=Average(peakVal)
            xy=zip([int(x) for x in xmap(xpositions)],[int(y) for y in ymap(ypositions)])
            #---------------------------------------------------------------------------
            #Matrix assignment

            for i,a,b,c in zip(xy,dopplers,valrange,peakVal):
                matrix[i[0]][i[1]]=a,b,c
                matrixav[alpha-1,frame]=AvPosx,AvPosy,AvDoppler,AvValrange,AvPeakVal
            #---------------------------------------------------------------------------
            #List of frames
            RadarFrameArray.append(matrix)
            OpticalFrameArray.append(opticalmotion[str(alpha)][frame])

        #-------------------------------------------------------------------------------
        #Motion

        RadarMotion = np.stack(RadarFrameArray,axis=0)
        OpticalMotion=np.stack(OpticalFrameArray,axis=0)
        RadarMotion=np.expand_dims(RadarMotion, axis=0)
        OpticalMotion = np.expand_dims(OpticalMotion, axis=0)

    folder="C:/Users/prav/Dropbox/source"
    PathMotionRadar=folder+"/Radar.npy"
    PathMotionOptical=folder+"/Optical.npy"
    PathMotionAverageRadar=folder+"/AverageRadar.npy"
    PathMotionLabel=folder+"/Labels.npy"




    try:

        #------------------------------------------------------------------
        #Radar Numpy Output: RNO

        RadarNumpyOutput=load(PathMotionRadar)
        RadarNumpyOutputConcatenate = np.concatenate((RadarNumpyOutput, RadarMotion), axis=0)
        save(PathMotionRadar, RadarNumpyOutputConcatenate)

        #-------------------------------------------------------------------
        # Optical Numpy Output: ONO

        OpticalNumpyOutput = load(PathMotionOptical)
        OpticalNumpyOutputConcatenate = np.concatenate((OpticalNumpyOutput, OpticalMotion), axis=0)
        save(PathMotionOptical, OpticalNumpyOutputConcatenate)
        #-------------------------------------------------------------------
        # Average Radar Output: AO

        AverageRadarOutput = load(PathMotionAverageRadar)
        AverageRadarOutputConcatenate = np.concatenate((AverageRadarOutput, matrixav), axis=0)
        save(PathMotionAverageRadar, AverageRadarOutputConcatenate)
        #-------------------------------------------------------------------
        # Label Numpy Output: LNO


        LabelNumpyOutput = load(PathMotionLabel)
        LabelNumpyOutputConcatenate = np.concatenate((LabelNumpyOutput, label), axis=0)
        save(PathMotionLabel, LabelNumpyOutputConcatenate)
        #-------------------------------------------------------------------



        print(RadarNumpyOutputConcatenate.shape)
        print(OpticalNumpyOutputConcatenate.shape)
        print(AverageRadarOutputConcatenate.shape)
        print(LabelNumpyOutputConcatenate.shape)


    except:

        save(PathMotionRadar , RadarMotion)
        save(PathMotionOptical, OpticalMotion)
        save(PathMotionAverageRadar, matrixav)
        save(PathMotionLabel , label)

        print(RadarMotion.shape)
        print(OpticalMotion.shape)
        print(matrixav.shape)
        print(label.shape)





for someval in range(0,5):

    run()
    time.sleep(2)

print (label)


folder="C:/Users/prav/Dropbox/source"
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged
OpticalData=load(folder+"/Optical.npy")
RadarData=load(folder+"/Radar.npy")
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
np.save(folder+'/ThreshGrayOpticalData.npy',output)
#
