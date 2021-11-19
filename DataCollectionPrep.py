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
np.set_printoptions(threshold=np.inf)
inputsamples=1
inputname=4
motionname='Hand2'
CI=30
label=np.array([inputname]*inputsamples)
def run():
    motion,name,opticalmotion=read.perform(inputname,inputsamples,CI)

    xmap = interp1d([-2,2],[0,99],bounds_error=False,fill_value=(0,99),kind='linear')
    ymap = interp1d([0,1.5],[0,99],bounds_error=False,fill_value=(0,99),kind='linear')

    RadarMotionArray=[]
    OpticalMotionArray=[]
    matrixav=np.zeros((inputsamples,CI,5))
    for alpha in range(1,inputsamples+1):

        RadarFrameArray=[]
        OpticalFrameArray=[]
        for frame in range(0,CI):
            matrix = np.zeros((100, 100,3))
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
            OpticalFrameArray.append(opticalmotion[str(alpha)][frame][1])

        #-------------------------------------------------------------------------------
        #Motion

        RadarMotion = np.stack(RadarFrameArray,axis=0)
        OpticalMotion=np.stack(OpticalFrameArray,axis=0)
        RadarMotion=np.expand_dims(RadarMotion, axis=0)
        OpticalMotion = np.expand_dims(OpticalMotion, axis=0)





    try:

        #------------------------------------------------------------------
        #Radar Numpy Output: RNO

        RadarNumpyOutput=load("motions\\Radar.npy")
        RadarNumpyOutputConcatenate = np.concatenate((RadarNumpyOutput, RadarMotion), axis=0)
        save("motions\\Radar.npy", RadarNumpyOutputConcatenate)

        #-------------------------------------------------------------------
        # Optical Numpy Output: ONO

        OpticalNumpyOutput = load("motions\Optical.npy")
        OpticalNumpyOutputConcatenate = np.concatenate((OpticalNumpyOutput, OpticalMotion), axis=0)
        save("motions\Optical.npy", OpticalNumpyOutputConcatenate)
        #-------------------------------------------------------------------
        # Average Radar Output: AO

        AverageRadarOutput = load("motions\AverageRadar.npy")
        AverageRadarOutputConcatenate = np.concatenate((AverageRadarOutput, matrixav), axis=0)
        save("motions\AverageRadar.npy", AverageRadarOutputConcatenate)
        #-------------------------------------------------------------------
        # Label Numpy Output: LNO


        LabelNumpyOutput = load("motions\Labels.npy")
        LabelNumpyOutputConcatenate = np.concatenate((LabelNumpyOutput, label), axis=0)
        save("motions\Labels.npy", LabelNumpyOutputConcatenate)
        #-------------------------------------------------------------------



        print(RadarNumpyOutputConcatenate.shape)
        print(OpticalNumpyOutputConcatenate.shape)
        print(AverageRadarOutputConcatenate.shape)
        print(LabelNumpyOutputConcatenate.shape)


    except:

        save("motions\\Radar.npy" , RadarMotion)
        save("motions\Optical.npy", OpticalMotion)
        save("motions\AverageRadar.npy", matrixav)
        save("motions\Labels.npy" , label)

        print(RadarMotion.shape)
        print(OpticalMotion.shape)
        print(matrixav.shape)
        print(label.shape)



for someval in range(0,1):
    time.sleep(3)
    run()
