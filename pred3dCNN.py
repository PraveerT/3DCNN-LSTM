import numpy as np
import time
from keras.models import load_model
from radar import read
from scipy.interpolate import interp1d

def Average(lst):
    return sum(lst) / len(lst)
pred=0
history = load_model('Optical.h5')
inputsamples=1
inputname=1
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
            OpticalFrameArray.append(opticalmotion[str(alpha)][frame])

        #-------------------------------------------------------------------------------
        #Motion

        RadarMotion = np.stack(RadarFrameArray,axis=0)
        OpticalMotion=np.stack(OpticalFrameArray,axis=0)
        RadarMotion=np.expand_dims(RadarMotion, axis=0)
        OpticalMotion = np.expand_dims(OpticalMotion, axis=0)


        predict_x = history.predict(np.transpose(OpticalMotion, (0,2,3,1,4)))
        classes_x = np.argmax(predict_x, axis=1)
        print(classes_x)
        listofword = ['youtube', 'scroll down']
        for i in classes_x:
            print (listofword[i])



while True:

    run()


