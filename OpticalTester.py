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
inputname=0

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





    try:

        #------------------------------------------------------------------
        #Radar Numpy Output: RNO

        RadarNumpyOutput=load("Test\\Radar.npy")
        RadarNumpyOutputConcatenate = np.concatenate((RadarNumpyOutput, RadarMotion), axis=0)
        save("Test\\Radar.npy", RadarNumpyOutputConcatenate)

        #-------------------------------------------------------------------
        # Optical Numpy Output: ONO

        OpticalNumpyOutput = load("C:\\Users\prav\Dropbox\source\OpticalTest.npy")
        OpticalNumpyOutputConcatenate = np.concatenate((OpticalNumpyOutput, OpticalMotion), axis=0)
        save("C:\\Users\prav\Dropbox\source\OpticalTest.npy", OpticalNumpyOutputConcatenate)
        #-------------------------------------------------------------------
        # Average Radar Output: AO

        AverageRadarOutput = load("Test\AverageRadar.npy")
        AverageRadarOutputConcatenate = np.concatenate((AverageRadarOutput, matrixav), axis=0)
        save("Test\AverageRadar.npy", AverageRadarOutputConcatenate)
        #-------------------------------------------------------------------
        # Label Numpy Output: LNO


        LabelNumpyOutput = load("Test\Labels.npy")
        LabelNumpyOutputConcatenate = np.concatenate((LabelNumpyOutput, label), axis=0)
        save("Test\Labels.npy", LabelNumpyOutputConcatenate)
        #-------------------------------------------------------------------



        print(RadarNumpyOutputConcatenate.shape)
        print(OpticalNumpyOutputConcatenate.shape)
        print(AverageRadarOutputConcatenate.shape)
        print(LabelNumpyOutputConcatenate.shape)


    except:

        save("Test\\Radar.npy" , RadarMotion)
        save("C:\\Users\prav\Dropbox\source\OpticalTest.npy", OpticalMotion)
        save("Test\AverageRadar.npy", matrixav)
        save("Test\Labels.npy" , label)

        print(RadarMotion.shape)
        print(OpticalMotion.shape)
        print(matrixav.shape)
        print(label.shape)


while True:
    run()
    time.sleep(10)
