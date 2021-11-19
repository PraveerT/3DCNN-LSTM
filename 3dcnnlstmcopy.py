import pandas as pd
import random
import numpy as np
from numpy import load
import os
import zipfile
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential,Model
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D,Activation,Dropout,Concatenate
img_rows, img_cols, img_depth =100, 100, 100
input_shape = (2, 100,100, 100, 1)
data=load("motions\Hand2.npy")
data=np.transpose(data, (0,2,3,1))
labels=load("motions\labelsHand2.npy")

print (labels)
print(data.shape,labels.shape)
y=np.expand_dims(data,axis=4)
data_train, data_test, labels_train, labels_test = train_test_split(y,labels)
model = Sequential()
model.add(Conv3D(6, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=y.shape[1:]))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Conv3D(6, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Flatten())

model2 = Sequential()
#input
model2.add(Dense(32, input_shape=((30,1))))
model2.add(Activation("elu"))
model2.add(Dropout(0.5))
model2.add(Dense(16))
model2.add(Activation("elu"))
model2.add(Dropout(0.25))
model2.add(Flatten())



merged = Concatenate()([model.output,model2.output])
z = Dense(64, activation='relu', kernel_initializer='he_uniform')(merged)
z = Dense(3, activation='softmax')(z)


modelo = Model(inputs=[model.input, model2.input], outputs=z)
# Build model.


# Compile model.
initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96
    , staircase=True
)
opt=tf.keras.optimizers.Adam(lr=1e-3,decay=1e-5,clipvalue=0.5)
modelo.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
)

print (data_train[:,:,:,:].shape,"shape")
history=modelo.fit((data_train[:,:,30,1],data_train),
                   (labels_train,labels_train),
          epochs=20,
          validation_data=(data_test, labels_test))


history.model.save('3DCNN_lstm_model4.h5')