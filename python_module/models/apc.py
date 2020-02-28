"""
@date 26.02.2020
Autoregressive Predictive Coding model
[An unsupervised autoregressive model for speech representation learning]

This corresponds to a translation from the Pytorch implementation
(https://github.com/iamyuanchung/Autoregressive-Predictive-Coding) to Keras implementation
"""
import os
from datetime import datetime

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Input, Dense, Dropout, GRU, Add, Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from models.model_base import ModelBase

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


class APCModel(ModelBase):

    def load_prediction_configuration(self, config):
        """
        It uses implementation from ModelBase
        :param config: dictionary with the configuration for predictions
        :return: the instance model have all configuration parameters from config
        """
        super(APCModel, self).load_prediction_configuration(config)

    def train(self):
        """
        Train APC model, optimiser adam and loss L1 (mean absolute error)
        :return: an APC trained model. The model is also saved in the specified folder (output_path param in the
                 training configuration)
        """
        # Configuration of learning process
        adam = Adam(lr=0.001)
        self.model.compile(optimizer=adam, loss='mean_absolute_error')

        # Callbacks for training
        # Adding early stop based on validation loss and saving best model for later prediction
        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=self.early_stop_epochs)
        checkpoint = ModelCheckpoint(os.path.join(self.full_path_output_folder, self.language +
                                                  datetime.now().strftime("_%Y_%m_%d-%H_%M") + '.h5'),
                                     monitor='val_loss', mode='min', verbose=1, save_best_only=True)

        # Tensorboard
        log_dir = os.path.join(self.logs_folder_path, datetime.now().strftime("%Y_%m_%d-%H_%M"))
        tensorboard = TensorBoard(log_dir=log_dir, write_graph=True, profile_batch=0)

        # Train the model
        self.model.fit(self.x_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.3,
                       callbacks=[tensorboard, early_stop, checkpoint])

        return self.model

    def predict(self, x_test, x_test_ind, duration):
        """
        It uses implementation from ModelBase
        :param x_test: test input features
        :param x_test_ind: indices mapping
        :param duration: string that states the duration in seconds of test files
        :return: predictions files
        """
        super(APCModel, self).predict(x_test, x_test_ind, duration)

    def load_training_configuration(self, config, x_train, y_train):
        """
        It loads configuration from dictionary, and instantiates the model architecture
        :param config: a dictionary with the configuration parameters for training
        :param x_train: a numpy array with the input features
        :param y_train: a numpy array with the output features
        :return: an instance will have the parameters from configuration and the model architecture
        """
        super(APCModel, self).load_training_configuration(config, x_train, y_train)

        # Model architecture: PreNet (stacked linear layers with ReLU activation and dropout) ->
        # ACP (multi-layer GRU network) -> Postnet(Conv1D this is only used during training)

        # Load apc configuration parameters
        apc_config = config['model']['apc']
        self.prenet = apc_config['prenet']
        self.prenet_layers = apc_config['prenet_layers']
        self.prenet_dropout = apc_config["prenet_dropout"]
        self.prenet_units = apc_config["prenet_units"]
        self.rnn_layers = apc_config["rnn_layers"]
        self.rnn_dropout = apc_config["rnn_dropout"]
        self.rnn_units = apc_config["rnn_units"]
        self.residual = apc_config["residual"]

        # input size and number of features. Input is numpy array of size (samples, time-steps, features)
        self.input_shape = self.x_train.shape[1:]
        self.features = self.x_train.shape[2]

        # Input tensor
        input_feats = Input(shape=self.input_shape, name='input_layer')

        # PreNet
        rnn_input = input_feats
        if self.prenet:
            for i in range(self.prenet_layers):
                rnn_input = Dense(self.prenet_units, activation='relu', name='prenet_linear_' + str(i))(rnn_input)
                rnn_input = Dropout(self.prenet_dropout, name='prenet_dropout_'+str(i))(rnn_input)

        # RNN
        for i in range(self.rnn_layers):
            # TODO Padding for sequences is not yet implemented
            if i+1 < self.rnn_layers:
                # All GRU layers will have rnn_units units except last one
                rnn_output = GRU(self.rnn_units, return_sequences=True, name='rnn_layer_'+str(i))(rnn_input)
            else:
                # Last GRU layer will have latent_dimension units
                if self.residual and self.latent_dimension == self.rnn_units:
                    # The latent representation will be then the output of the residual connection.
                    rnn_output = GRU(self.latent_dimension, return_sequences=True, name='rnn_layer_'+str(i))(rnn_input)
                else:
                    rnn_output = GRU(self.latent_dimension, return_sequences=True, name='latent_layer')(rnn_input)

            if i+1 < self.rnn_layers:
                # Dropout to all layers except last layer
                rnn_output = Dropout(self.rnn_dropout, name='rnn_dropout_'+str(i))(rnn_output)

            if self.residual:
                # residual connection is applied to last layer if the latent dimension and rnn_units are the same,
                # otherwise is omitted. And to the first layer if the PreNet units and RNN units are the same,
                # otherwise is omitted also for first layer.
                residual_last = (i+1 == self.rnn_layers and self.latent_dimension == self.rnn_units)
                residual_first = (i == 0 and self.prenet_units == self.rnn_units)

                if (i+1 < self.rnn_layers and i != 0) or residual_first:
                    rnn_input = Add(name='rnn_residual_'+str(i))([rnn_input, rnn_output])

                # Update output for next layer (PostNet) if this is the last layer. This will also be the latent
                # representation.
                if residual_last:
                    rnn_output = Add(name='latent_layer')([rnn_input, rnn_output])

                # Update input for next layer
                if not residual_first and i == 0:
                    # Residual connection won't be applied but we need to update input value for next RNN layer
                    # to the output of RNN + dropout
                    rnn_input = rnn_output
            else:
                # Output of the dropout or RNN layer in the case of the last layer
                rnn_input = rnn_output

        # PostNet
        postnet_layer = Conv1D(self.features, kernel_size=1, padding='same', name='postnet_conv1d')(rnn_output)

        # APC Model
        self.model = Model(input_feats, postnet_layer)
