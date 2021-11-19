import pandas as pd
import random
import numpy as np
import os
import csv
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from ann_visualizer.visualize import ann_viz
def readData():

    labels = pd.read_csv('label.csv', header = None)

    labels = labels.values
    labels = labels - 1

    print('One Hot Encoding Data...')
    #
    datax = pd.read_csv('avx.csv', header = None)
    datax=datax.values
    datay = pd.read_csv('avy.csv', header=None)
    datay = datay.values
    datad = pd.read_csv('avdoppler.csv', header=None)
    datad = datad.values
    datar = pd.read_csv('avrange.csv', header=None)
    datar = datar.values
    datap = pd.read_csv('avpeakval.csv', header=None)
    datap = datap.values
    #
    newdata=np.dstack((datax, datay,datad,datar,datap))
    # newdata=datax.reshape(len(datap),100,1)




    return labels,newdata


print('Reading data...')
labels,data= readData()
print (data.shape)

print('Splitting Data')
data_train, data_test, labels_train, labels_test = train_test_split(data,labels)

print('Building Model...')
# Create model
model = Sequential()
model.add(LSTM(128, input_shape=(data_train.shape[1:]),recurrent_dropout=0.2, activation='relu', return_sequences=True))


model.add(LSTM(2, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))
opt=tf.keras.optimizers.Adam(lr=1e-3,decay=1e-5,clipvalue=0.5)
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
)
history=model.fit(data_train,
          labels_train,
          epochs=300,
          validation_data=(data_test, labels_test))


history.model.save('lstm_model.h5')
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()