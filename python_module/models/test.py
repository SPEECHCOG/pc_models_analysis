import os
from datetime import datetime

import numpy as np
from sklearn.decomposition import PCA
from tensorflow import keras

import tensorflow as tf
from tensorflow.keras.losses import mean_absolute_error, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Input, Dense, Dropout, GRU, Add, Conv1D, MaxPooling1D, Lambda
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

from keras import backend as K

from models.create_prediction_files import create_prediction_files
from read_configuration import read_configuration_json, load_training_features, load_test_set

hidden = [32,16,8,16,32]
learning_rate = 0.001
steps = 5

config = read_configuration_json('../config_test.json', True, False)['training']
# Obtain input/output features to train the model
x_train, y_train = load_training_features(config['train_in'], config['train_out'])

input_shape = x_train.shape[1:]
features = x_train.shape[2]

# Input tensor
input_feats = Input(shape=input_shape, name='input_layer')
input_feats_future = Input(shape=input_shape, name='input_fut_layer')

#conv_layer = Conv1D(units, kernel_size=3, padding='same', name='conv1d')(input_feats)
#max_pooling = MaxPooling1D(3,1,padding='same', name='pool1')(conv_layer)

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

conv_layers = []
maxpool_layers = []

for i in range(len(hidden)):
    conv_layers.append(Conv1D(hidden[i], kernel_size=3, padding='same', activation='relu',
                              name='conv_' + str(i)))
    if i == len(hidden)-1:
        maxpool_layers.append(MaxPooling1D(3, 1, padding='same', name='latent_layer'))
    else:
        maxpool_layers.append(MaxPooling1D(3, 1, padding='same', name='pool_' + str(i)))

# Use for restricting latent representations
autoencoder = Conv1D(features, kernel_size=1, strides=1, padding='same', name='autoencoder')

# Predictive coding part
latent_layers = []
maxpool_latent_layers = []

for i in range(len(hidden)):
    latent_layers.append(Conv1D(hidden[i], kernel_size=3, padding='same', activation='relu', name='conv_fut_'+str(i)))
    maxpool_latent_layers.append(MaxPooling1D(3,1,padding='same', name='latent_pool_'+str(i)))

#future_layer = Conv1D(hidden[-1], kernel_size=5, padding='same', activation='relu', name='gru')

# Outputs
for i in range(len(conv_layers)):
    if i == 0:
        # Firt layer should be applied to inputs
        future_prediction = maxpool_layers[i](conv_layers[i](input_feats))
        latent_future = maxpool_layers[i](conv_layers[i](input_feats_future))
        autoencoder_prediction = maxpool_layers[i](conv_layers[i](input_feats))
    else:
        future_prediction = maxpool_layers[i](conv_layers[i](future_prediction))
        latent_future = maxpool_layers[i](conv_layers[i](latent_future))
        autoencoder_prediction = maxpool_layers[i](conv_layers[i](autoencoder_prediction))

for i in range(len(latent_layers)):
    future_prediction = maxpool_latent_layers[i](latent_layers[i](future_prediction))


autoencoder_prediction = autoencoder(autoencoder_prediction)

model_out = keras.layers.Concatenate()([future_prediction, latent_future])


def final_loss_func(args):
    auto_true, auto_pred, latent_true, latent_pred = args
    auto_mae = K.mean(K.abs((auto_pred - auto_true)), keepdims=True)
    latent_mae = K.mean(K.abs(latent_pred - latent_true), keepdims=True)
    return K.abs(auto_mae - latent_mae)


def contrastive_loss_func(args):
    latent_true, latent_pred = args
    true_futures = latent_true[0, :, :]
    predictions = []
    # True predictions:
    predictions.append(latent_pred[0, :, :])

    for i in range(steps - 1):
        predictions.append(latent_true[i+1, :, :])

    dist_correct = K.clip(K.exp(K.sum((predictions[0] * true_futures), axis=-1)), 1e-6, 1e6)
    dist_false = []

    for i in range(steps - 1):
        dist_false.append(K.clip(K.exp(K.sum((predictions[0] * predictions[i+1]), axis=-1)), 1e-6, 1e6))

    dist_false.append(dist_correct)
    total_dist_false = tf.add_n(dist_false)

    loss = -K.mean(K.log(dist_correct) - K.log(total_dist_false), keepdims=True)
    return loss

#final_loss = Lambda(final_loss_func, output_shape=(1,), name='final_loss')([input_feats, autoencoder_prediction,
#                                                                      latent_future, future_prediction])


final_loss = Lambda(contrastive_loss_func, output_shape=(1,), name='final_loss')([latent_future, future_prediction])

model = Model(inputs=[input_feats,input_feats_future], outputs=[final_loss, autoencoder_prediction])

print(model.summary())


half = hidden[-1]

def mae_latent(y_true,y_pred):
    mae = K.mean(K.abs((y_pred[:, :, 0:half] - y_pred[:, :, half:])))
    return mae

adam = Adam(lr=learning_rate)

model.compile(optimizer=adam, loss={'final_loss': lambda y_true, y_pred: y_pred, 'autoencoder': 'mean_absolute_error'})


#y_dummy = np.random.rand(x_train.shape[0],200,units*2)
y_dummy = np.random.rand(x_train.shape[0], 1, 1)

log_dir = 'logs/' + 'steps_' + str(steps)
tensorboard = TensorBoard(log_dir=log_dir, write_graph=True, profile_batch=0)

y_train_10 = np.roll(y_train.reshape(y_train.shape[0]*y_train.shape[1], y_train.shape[-1]),
                     -10, axis=0).reshape(y_train.shape)

model.fit(x=[x_train,y_train], y=[y_dummy, x_train], batch_size=32, epochs=200, verbose=1,
          validation_split=0.3,
          callbacks=[tensorboard]
          )

_, auto_out = model.predict([x_train, x_train])



# model.save('convpc_future_decoding.h5')
#
# # Predict using this model
#
# #prediction = Model([input_feats, input_feats_future], model.get_layer('latent_layer').output)
# prediction = model
#
# config = read_configuration_json('../config_test.json', False, True)['prediction']
# x_test, x_test_ind = load_test_set(config['test_set'], '10')
#
#
#
# _, predictions, _ = prediction.predict([x_test, x_test])
# predictions = predictions[:, :, :hidden[-1]]
#
# use_pca = True
# if use_pca:
#     pca = PCA(0.95)  # Keep components that coverage 95% of variance
#     pred_orig_shape = predictions.shape
#     predictions = predictions.reshape(-1, predictions.shape[-1])
#     predictions = pca.fit_transform(predictions)
#     pred_orig_shape = list(pred_orig_shape)
#     pred_orig_shape[-1] = predictions.shape[-1]
#     pred_orig_shape = tuple(pred_orig_shape)
#     predictions = predictions.reshape(pred_orig_shape)
#
#
# output_folder = config['output_path']
# model_path = config['model_path']
# model_folder_name = config['model_folder_name']
# type = config['model_type']
# features_folder_name = config['features_folder_name']
# language = config['language']
# use_last_layer = config['use_last_layer']
# window_shift = config['window_shift']
# files_limit = config['files_limit']
#
# # Create folder for predictions
# full_predictions_folder_path = os.path.join(output_folder, model_folder_name,
#                                             features_folder_name, language, ('10' + 's'))
# os.makedirs(full_predictions_folder_path, exist_ok=True)
#
# # Create predictions text files
# total_files = create_prediction_files(predictions, x_test_ind, full_predictions_folder_path, window_shift,
#                                       limit=files_limit)

