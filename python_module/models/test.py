import numpy as np
from tensorflow import keras

import tensorflow as tf
from tensorflow.keras.losses import mean_absolute_error
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Input, Dense, Dropout, GRU, Add, Conv1D, MaxPooling1D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

from keras import backend as K



units = 128
gru_units = 16
steps = 5
window = 3
learning_rate = 0.001

x_train_all = np.random.rand(1000,200,39)
x_train_in = x_train_all[:, :-steps, :]
x_train_future = x_train_all[:, steps:, :]


input_shape = x_train_in.shape[1:]
features = x_train_all.shape[2]



# Input tensor
input_feats = Input(shape=input_shape, name='input_layer')
input_feats_future = Input(shape=input_shape, name='input_fut_layer')

#conv_layer = Conv1D(units, kernel_size=3, padding='same', name='conv1d')(input_feats)
#max_pooling = MaxPooling1D(3,1,padding='same', name='pool1')(conv_layer)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


conv_layer = Conv1D(units, kernel_size=3, padding='same', name='conv1d')
max_pooling = MaxPooling1D(3,1,padding='same', name='pool1')

postnet = Conv1D(features, 1, 1, padding='same', name='convlast')

#gru = GRU(units, return_sequences=True, name='gru')(max_pooling[:, :-steps, :])
gru = GRU(units, return_sequences=True, name='gru')

future_out = max_pooling(conv_layer(input_feats_future))
prediction = gru(max_pooling(conv_layer(input_feats)))
short_prediction = postnet(max_pooling(conv_layer(input_feats)))


model_out = keras.layers.Concatenate()([prediction, future_out])

model = Model(inputs=[input_feats,input_feats_future], outputs=[model_out, short_prediction])

print(model.summary())

#model = Model(input_feats, gru)
#predictive_m = Model(input_feats, max_pooling)

half = units

def newloss(y_true,y_pred):
    print(y_pred)
    print(y_pred[:,:,0:half])
    print(y_pred[:,:,half:2*half])
    mae = mean_absolute_error(y_pred[:,:,0:half],y_pred[:,:,half:2*half])
    #mae = K.mean(K.square((y_pred[:, :, 0:half] - y_pred[:, :, half:2 * half])))
    return mae




adam = Adam(lr=learning_rate)
model.compile(optimizer=adam, loss=[newloss, 'mean_absolute_error'], loss_weights=[1.0,1.0])

x_dummy = np.random.rand(1000,1,1)

#log_dir = 'logs/'
#tensorboard = TensorBoard(log_dir=log_dir, write_graph=True, profile_batch=0)


model.fit([x_train_in,x_train_future],y=[x_dummy,x_train_in], batch_size=32, epochs=20, verbose=1,
          validation_split=0.2,
          #callbacks=[tensorboard]
          )

#
# def predictive_loss(input):
#     def loss(y_true, y_pred):
#         y_true = predictive_m.predict(input, steps=32)
#         print(y_true)
#         print(y_pred)
#         y_true = y_true[:, steps:, :]
#         return mean_absolute_error(y_true, y_pred)
#     return loss


#
# adam = Adam(lr=learning_rate)
# model.compile(optimizer=adam, loss=predictive_loss(input_feats))
#
# model.fit(x_train, x_train, batch_size=32, epochs=100, verbose=1, validation_split=0.2)

print(model.summary())


"""
for epoch in range(10):
    print('......................... epoch .............................' + str(epoch) )
    for i in range(5):
        print('......................... chunk .............................' + str(i+1))
        data_orderX,data_orderY = randOrder(n_train)
        Y_train = []
        X_train = []
        infile = open('/worktmp/khorrami/work/projects/project_4/outputs/step_4/tripletData/trainX' + str(5),'rb')
        X_train_temp = pickle.load(infile)
        infile.close()
        Y_train = Y_data[i][data_orderY]
        X_train = X_train_temp[data_orderX]
        history = model.fit([Y_train, X_train], bin_target_epoch, shuffle=False, epochs=1,batch_size=120,  
                            validation_data=([Y_val_triplet,X_val_triplet],binary_target_val))
        Y_train = []
        X_train = []
        val_epoch = history.history['val_loss'][0]
        train_epoch = history.history['loss'][0]
        allepochs_valloss.append(val_epoch)
        allepochs_trainloss.append(train_epoch)
        if val_epoch <= val_indicator:
            val_indicator = val_epoch
            weights = model.get_weights()
            model.set_weights(weights)
            model.save_weights('%s/v0_model_weights.h5' % modeldir)
            weights = []
            history = []
    print ('................................................................   Final val_loss after this epoch =    ' + 
            str(val_indicator))
    scipy.io.savemat('/worktmp/khorrami/work/projects/project_4/outputs/step_4/model/triplets/valtrainloss.mat',
                 {'allepochs_valloss':allepochs_valloss,'allepochs_trainloss':allepochs_trainloss})
# .............................................................................
"""