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
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D,Activation,Dropout,Concatenate,LSTM
img_rows, img_cols, img_depth =100, 100, 100
input_shape = (2, 100,100, 100, 3)
data=load("motions\Hand2.npy")
data=np.transpose(data, (0,2,3,1,4))
labels=load("motions\labelsHand2.npy")
data2=load("motions\LSTMdata.npy")
print (labels)
print(data.shape,labels.shape,data2.shape)

data_train, data_test, labels_train, labels_test,data_train2,data_test2 = train_test_split(data,labels,data2)

model = Sequential()
model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=data.shape[1:]))
model.add(MaxPooling3D(pool_size=(1, 2, 2))) #previously (2,2,2)
model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling3D(pool_size=(1, 2, 2)))
model.add(Flatten())


model2 = Sequential()
model2.add(LSTM(128, input_shape=(data_train2.shape[1:]),recurrent_dropout=0.2, activation='relu', return_sequences=True))


model2.add(LSTM(128, activation='relu'))
model2.add(Dropout(0.5))

model2.add(Dense(64, activation='relu'))
model2.add(Dropout(0.5))


merged = Concatenate()([model.output,model2.output])
z = Dense(128, activation="relu")(merged)
z = Dropout(0.4)(z)
z = Dense(1024, activation="relu")(z)
z = Dense(5, activation="softmax")(z)

model3 = Model(inputs=[model.input, model2.input], outputs=z)
model3.summary()

# Build model.


# Compile model.
initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96
    , staircase=True
)
opt=tf.keras.optimizers.Adam(learning_rate=1e-3,decay=1e-5,clipvalue=0.5)
model3.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
)


history=model3.fit([data_train,data_train2],
          labels_train,epochs=20,
          validation_data=([data_test,data_test2], labels_test))

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
history.model.save('youtube.h5)

