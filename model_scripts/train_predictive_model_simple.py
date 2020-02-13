import scipy,scipy.io


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.layers import Dense, merge
from keras.layers.merge import concatenate
from keras.layers import LSTM
import keras
import numpy

from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge, TimeDistributed, add
from keras.models import Model
from keras.layers.core import Masking

from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers import Dropout
from keras.layers.convolutional import UpSampling1D
from keras import optimizers
from keras.callbacks import Callback
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, UpSampling2D, Conv1D, UpSampling1D, MaxPooling1D, AveragePooling1D, Conv1D, Multiply, GRU

import os

datadir = '/Users/rasaneno/rundata/ZS2020_tmp/'
language = 'english'

loaddata = scipy.io.loadmat('%slogmel_concat_%s_in.mat' % (datadir,language))
X_in = loaddata['X_in']

loaddata = scipy.io.loadmat('%slogmel_concat_%s_out1.mat' % (datadir,language))
X_out1 = loaddata['X_out1']

loaddata = scipy.io.loadmat('%slogmel_concat_%s_out2.mat' % (datadir,language))
X_out2 = loaddata['X_out2']

loaddata = scipy.io.loadmat('%slogmel_concat_%s_out3.mat' % (datadir,language))
X_out3 = loaddata['X_out3']

model = Sequential()
sequence = Input(shape=(X_in.shape[1:]))

lstm_1 = LSTM(30,return_sequences=True)(sequence)
lstm_2 = LSTM(30,return_sequences=True)(lstm_1)

predictor1 = Dense(24,activation='linear')
td1 = TimeDistributed(predictor1)(lstm_1)

predictor2 = Dense(24,activation='linear')
td2 = TimeDistributed(predictor1)(lstm_2)

model = Model(outputs=[td1,td2], inputs=sequence)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
print(model.summary())
model.fit(X_in, [X_out1, X_out2], shuffle=True, epochs=5,batch_size=100, validation_split=0.1)

# Try Conv-variant that needs to maxpool in time at different resolutions

n_channels = 32

x = Conv1D(n_channels,3, activation='relu', padding='same')(sequence) # 1 frame ahead
pool1 = MaxPooling1D(3,1,padding='same')(x)
x2 = Conv1D(n_channels,3, activation='relu', padding='same')(pool1) # 2 frames ahead
pool2 = MaxPooling1D(3,1,padding='same')(x2)

x3 = Conv1D(n_channels,5, activation='relu', padding='same')(pool2) # 3 frames ahead
pool3 = MaxPooling1D(5,1,padding='same')(x3)

x4 = Conv1D(n_channels,15, activation='relu', padding='same')(pool3) # 3 frames ahead
pool4 = MaxPooling1D(15,1,padding='same')(x4)




predictor1 = Dense(24,activation='linear')
td1 = TimeDistributed(predictor1)(pool2)

predictor2 = Dense(24,activation='linear')
td2 = TimeDistributed(predictor2)(pool3)

predictor3 = Dense(24,activation='linear')
td3 = TimeDistributed(predictor3)(pool4)

model = Model(outputs=[td1,td2,td3], inputs=sequence)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
print(model.summary())
model.fit(X_in, [X_out1, X_out2, X_out3], shuffle=True, epochs=5,batch_size=100, validation_split=0.1)
