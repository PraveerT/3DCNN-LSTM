#motion #frame #val
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import pickle as Pickle
import random
import csv
import read
from scipy.optimize import curve_fit
#1 waves
#2 buttonpress
pred=0

inputsamples=10
inputname=2

motion,name=read.perform(inputname,inputsamples)

def Average(lst):
   return sum(lst) / len(lst)


a = open('avx.csv', 'a', newline='')
b = open('avy.csv', 'a', newline='')
c = open('avdoppler.csv', 'a', newline='')
d = open('avrange.csv', 'a', newline='')
e = open('avpeakval.csv', 'a', newline='')
f = open('label.csv', 'a', newline='')


writerx = csv.writer(a)
writery = csv.writer(b)
writerd = csv.writer(c)
writerr = csv.writer(d)
writerp = csv.writer(e)
writerlab = csv.writer(f)


# create the csv writer

fig, axs = plt.subplots(5)
fig.suptitle('button press')


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
    writerx.writerow(x)
    writery.writerow(y)
    writerd.writerow(doppler)
    writerr.writerow(Range)
    writerp.writerow(peakval)
    writerlab.writerow(str(inputname))



for valinlist in range(0,len(x)):

   if x[valinlist]==0 and valinlist>0:
       x[valinlist]=x[valinlist-1]


for valinlist in range(0,len(y)):
   if y[valinlist]==0.5 and valinlist>0:
       y[valinlist]=y[valinlist-1]

for valinlist in range(0,len(doppler)):
   if doppler[valinlist]==0 and valinlist>0:
       doppler[valinlist]=doppler[valinlist-1]

for valinlist in range(0,len(doppler)):
   if peakval[valinlist]==0 and valinlist>0:
       peakval[valinlist]=peakval[valinlist-1]

for valinlist in range(0,len(y)):
   if Range[valinlist]==0 and valinlist>0:
       Range[valinlist]=Range[valinlist-1]

axs[0].plot(x,color=(.2,.2,.2),label='x')
axs[1].plot(y,label='y')
axs[2].plot(doppler,label='doppler')
axs[3].plot(Range,label='range')
axs[4].plot(peakval,label='peakval')
axs[0].legend()
axs[1].legend()
axs[2].legend()
axs[3].legend()
axs[4].legend()


a.close()
b.close()
c.close()
d.close()
e.close()
f.close()



plt.show()