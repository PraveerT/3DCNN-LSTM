
import pandas as pd
import numpy as np
from keras.models import load_model
import read
model = load_model('lstm_model.h5')
inputname=1
inputsamples=1
motion,name=read.perform(inputname,inputsamples)

def Average(lst):
   return sum(lst) / len(lst)




def imp(val,d,l):

    label=l.values[val]
    data = d.values[val]
    data = data.reshape((1,100,1))




    # make predictions
    predict_x=model.predict(data)
    classes_x = np.argmax(predict_x, axis=1)
    if classes_x == 1:
        print("Button Press")

    if classes_x == 0:
        print("Waves")


for alpha in range(1,inputsamples+1):
    x = []
    y = []
    doppler = []
    Range = []
    peakval = []
    for i in range (0,100):


        x.append(Average(motion[str(alpha)][i]['x']) * len(motion[str(alpha)][i]['x']))
        y.append(Average(motion[str(alpha)][i]['y']) * len(motion[str(alpha)][i]['y']))
        doppler.append(Average(motion[str(alpha)][i]['doppler']) * len(motion[str(alpha)][i]['doppler']))
        Range.append(Average(motion[str(alpha)][i]['range']) * len(motion[str(alpha)][i]['range']))
        peakval.append(Average(motion[str(alpha)][i]['peakVal']))
newdata=np.dstack((x,y,doppler,Range,peakval))
predict_x=model.predict(newdata)
classes_x = np.argmax(predict_x, axis=1)

print(predict_x,classes_x)

if classes_x==0:
    print("inout")

if classes_x==1:
    print ("waves")