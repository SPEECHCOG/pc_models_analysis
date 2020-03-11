import numpy as np
from tensorflow import keras

import tensorflow as tf
from tensorflow.keras.losses import mean_absolute_error, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Input, Dense, Dropout, GRU, Add, Conv1D, MaxPooling1D, Lambda
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

from keras import backend as K

from read_configuration import read_configuration_json, load_training_features

units = 8
gru_units = 16
steps = 5
window = 3
learning_rate = 0.001

config = read_configuration_json('../config_test.json', True, False)['training']
# Obtain input/output features to train the model
x_train, y_train = load_training_features(config['train_in'], config['train_out'])

# x_train = x_train[:2, :, :]
# y_train = y_train[:2, :, :]


input_shape = x_train.shape[1:]
features = x_train.shape[2]

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


conv_layer = Conv1D(units, kernel_size=3, padding='causal', activation='relu', name='conv1d')
max_pooling = MaxPooling1D(3,1,padding='same', name='pool1')

postnet = Conv1D(features, 1, 1, padding='causal', name='convlast')

#gru = GRU(units, return_sequences=True, name='gru')(max_pooling[:, :-steps, :])
latent_fut = Conv1D(units, kernel_size=5, padding='causal', activation='relu' , name='latent_fut')

future_out = max_pooling(conv_layer(input_feats_future))
prediction = latent_fut(max_pooling(conv_layer(input_feats)))
short_prediction = postnet(max_pooling(conv_layer(input_feats)))


model_out = keras.layers.Concatenate()([prediction, future_out])


def final_loss(args):
    auto_true, auto_pred, latent_true, latent_pred = args
    auto_mae = K.mean(K.abs((auto_pred - auto_true)), keepdims=True)
    latent_mae = K.mean(K.abs(latent_pred - latent_true), keepdims=True)
    print_op = tf.print("latent_mae: ", latent_mae)
    print_op1 = tf.print("auto_mae", auto_mae)

    # The idea is to minimise the difference between both losses as well as each loss. And to restrict constant values
    # by using the weights for losses. In my experiments that has help at least to have better latent representations
    with tf.control_dependencies([print_op, print_op1]):
        return K.abs(auto_mae - latent_mae)


loss_out = Lambda(final_loss, output_shape=(1,), name='final_loss')([input_feats, short_prediction,
                                                                     future_out, prediction])


model = Model(inputs=[input_feats,input_feats_future], outputs=[loss_out, short_prediction, model_out])

print(model.summary())

#model = Model(input_feats, gru)
#predictive_m = Model(input_feats, max_pooling)

half = units

def newloss2(y_true, y_pred):
    print_op = tf.print("autoencoder: y_true: ", y_true)
    print_op1 = tf.print("autoencoder: y_pred", y_pred)
    with tf.control_dependencies([print_op, print_op1]):
        mae = K.mean(K.abs((y_pred - y_true)))
    return mae

def newloss(y_true,y_pred):
    #mae = mean_squared_error(y_pred[:,:,0:half],y_pred[:,:,half:])
    #mae = K.mean(K.square((y_pred[:, :, 0:half] - y_pred[:, :, half:])))
    print_op = tf.print("latent: y_true: ", y_pred[:, :, half:])
    print_op1 = tf.print("latent: y_pred", y_pred[:, :, 0:half])
    with tf.control_dependencies([print_op, print_op1]):
        mae = K.mean(K.abs((y_pred[:, :, 0:half] - y_pred[:, :, half:])))
    return mae

adam = Adam(lr=learning_rate)
model.compile(optimizer=adam, loss={'final_loss':lambda y_true, y_pred: y_pred, 'convlast': 'mean_absolute_error',
                                    'concatenate': newloss}, loss_weights=[0.5, 0.4, 0.1])


#y_dummy = np.random.rand(x_train.shape[0],200,units*2)
y_dummy = np.random.rand(x_train.shape[0], 1, 1)

log_dir = 'logs/'
tensorboard = TensorBoard(log_dir=log_dir, write_graph=True, profile_batch=0)


model.fit(x=[x_train,y_train],y=[y_dummy, x_train, y_dummy], batch_size=32, epochs=100, verbose=1,
          validation_split=0.2,
          callbacks=[tensorboard]
          )

predictor = Model([input_feats, input_feats_future], postnet(max_pooling(conv_layer(input_feats))))

out = predictor.predict([x_train, y_train])

print(K.mean(K.abs((out - x_train))))
print(model.summary())
