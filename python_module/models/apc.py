"""
@date 26.02.2020
Autoregressive Predictive Coding model
[An unsupervised autoregressive model for speech representation learning]

This corresponds to a translation from the Pytorch implementation
(https://github.com/iamyuanchung/Autoregressive-Predictive-Coding) to Keras implementation
"""
from keras.layers import Input, Dense, Dropout, GRU, Add, Conv1D
from keras.models import Model

from models.model_base import ModelBase


class APCModel(ModelBase):

    def load_prediction_configuration(self, config):
        pass

    def train(self):
        pass

    def predict(self, x_test, x_test_ind, duration):
        pass

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
                rnn_input = Dense(self.prenet_units, activation='ReLU', name='prenet_linear_' + str(i))(rnn_input)
                rnn_input = Dropout(self.prenet_dropout, name='prenet_dropout_'+str(i))(rnn_input)

        # RNN
        for i in range(self.rnn_layers):
            # TODO Padding for sequences is not yet implemented
            if i+1 < self.rnn_layers:
                # All GRU layers will have rnn_units units
                rnn_output = GRU(self.rnn_units, name='rnn_layer_'+str(i))(rnn_input)
            else:
                # Last GRU layer will have latent_dimension units
                rnn_output = GRU(self.latent_dimension, name='latent_layer')(rnn_input)

            if i+1 < self.rnn_layers:
                # Dropout to all layers except last layer
                rnn_output = Dropout(self.rnn_dropout, name='rnn_dropout_'+str(i))(rnn_input)

            if self.residual:
                residual_last = (i+1 == self.rnn_layers and self.latent_dimension == self.rnn_units)
                # residual connection is applied to last layer if the latent dimension and rnn_units are the same,
                # otherwise is omitted.
                if residual_last or (i+1 < self.rnn_layers):
                    rnn_input = Add(name='rnn_residual_'+str(i))([rnn_input, rnn_output])
            else:
                # Output of the dropout or RNN layer in the case of the last layer
                rnn_input = rnn_output

        # The last rnn_input is the last output of the RNN
        rnn_output = rnn_input

        # PostNet
        postnet_layer = Conv1D(self.features, kernel_size=1, padding='same', name='postnet_conv1d')(rnn_output)

        # APC Model
        self.model = Model(input_feats, postnet_layer)
