#motion #frame #val
import numpy as np
from numpy import save,load
import time
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import pickle as Pickle
import random
import csv
import read
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
    motion,name=read.perform(inputname,inputsamples,CI)

    xmap = interp1d([-2,2],[0,99],bounds_error=False,fill_value=(0,99),kind='linear')
    ymap = interp1d([0,1.5],[0,99],bounds_error=False,fill_value=(0,99),kind='linear')

    BATARR=[]
    matrixlstm=np.zeros((inputsamples,CI,5))
    for alpha in range(1,inputsamples+1):

        arr=[]
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
            for i,a,b,c in zip(xy,dopplers,valrange,peakVal):
                matrix[i[0]][i[1]]=a,b,c
                matrixlstm[alpha-1,frame]=AvPosx,AvPosy,AvDoppler,AvValrange,AvPeakVal
            arr.append(matrix)

        motionARR = np.stack(arr,axis=0)
        BATARR.append(motionARR)
    OUTPUTARR=np.stack(BATARR,axis=0)


    try:
        OA=load("motions\Hand2.npy")
        LOA = load("motions\labelsHand2.npy")
        OUTCONC=np.concatenate((OA, OUTPUTARR), axis=0)
        save("motions\%s.npy" % motionname, OUTCONC)
        OUTCONCLABEL = np.concatenate((LOA, label), axis=0)
        save("motions\labels%s.npy" % motionname, OUTCONCLABEL)
        LSTMOA=load("motions\LSTMdata.npy")
        OUTLSTMCONC=np.concatenate((LSTMOA,matrixlstm),axis=0)
        save("motions\LSTMdata.npy",OUTLSTMCONC)
        print(OUTCONC.shape)
        print(OUTCONCLABEL.shape)
        print(OUTLSTMCONC.shape)

    except:

        save("motions\%s.npy" % motionname, OUTPUTARR)
        save("motions\labels%s.npy" % motionname, label)
        save("motions\LSTMdata.npy",matrixlstm)
        print (OUTPUTARR.shape)
        print (matrixlstm.shape)


for someval in range(0,10):
    time.sleep(3)
    run()
