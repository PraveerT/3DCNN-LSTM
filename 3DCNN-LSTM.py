from numpy import load
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential,Model
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D,Activation,Dropout,Concatenate,LSTM

#input_shape = (sample,60 ,100, 100, 3)

OpticalData=np.transpose(load("motions\Optical.npy"), (0,2,3,1,4))
RadarData=np.transpose(load("motions\Radar.npy"), (0,2,3,1,4))
AverageRadarData=load("motions\AverageRadar.npy")
Labels=load("motions\Labels.npy")

print(OpticalData.shape,RadarData.shape,AverageRadarData.shape,Labels.shape)

Optical_train, Optical_test,Average_train,Average_test , Labels_train, Labels_test=train_test_split(OpticalData,AverageRadarData,Labels)

TDCNNmodel = Sequential()
TDCNNmodel.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=OpticalData.shape[1:]))
TDCNNmodel.add(MaxPooling3D(pool_size=(1, 2, 2))) #previously (2,2,2)
TDCNNmodel.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'))
TDCNNmodel.add(MaxPooling3D(pool_size=(1, 2, 2)))
TDCNNmodel.add(Flatten())


LSTMmodel = Sequential()
LSTMmodel.add(LSTM(128, input_shape=(Average_train.shape[1:]),recurrent_dropout=0.2, activation='relu', return_sequences=True))


LSTMmodel.add(LSTM(128, activation='relu'))
LSTMmodel.add(Dropout(0.5))
LSTMmodel.add(Dense(64, activation='relu'))
LSTMmodel.add(Dropout(0.5))


merged = Concatenate()([TDCNNmodel.output,LSTMmodel.output])
z = Dense(128, activation="relu")(merged)
z = Dropout(0.4)(z)
z = Dense(1024, activation="relu")(z)
z = Dense(5, activation="softmax")(z)

Mergemodel = Model(inputs=[TDCNNmodel.input, LSTMmodel.input], outputs=z)
Mergemodel.summary()

# Build model.


# Compile model.
initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96
    , staircase=True
)
opt=tf.keras.optimizers.Adam(learning_rate=1e-3,decay=1e-5,clipvalue=0.5)
Mergemodel.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
)


history=Mergemodel.fit([Optical_train,Average_train],
          Labels_train,epochs=20,batch_size=1,
          validation_data=([Optical_test,Average_test], Labels_test))

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
history.model.save('Optical_Average.h5')
