"""
    @author Maria Andrea Cruz Blandon
    @date 05.03.2020

    Convolutional Predictive Coding:  In this model we use the schematic idea of Contrastive Predictive Coding in which
    we learn short-term representation and context-term representations, and ideas of Autoregressive Predictive Coding
    for calculate predictions of context-term representations.
"""
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Input, Dense, Dropout, GRU, Add, Conv1D, MaxPooling1D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

from models.model_base import ModelBase


class ConvPC(ModelBase):
    def load_training_configuration(self, config, x_train, y_train):
        """

        :param config:
        :param x_train:
        :param y_train:
        :return:
        """
        super(ConvPC, self).load_training_configuration(config, x_train, y_train)

        # Define model
        # input size and number of features. Input is numpy array of size (samples, time-steps, features)
        self.input_shape = self.x_train.shape[1:]
        self.features = self.x_train.shape[2]

        # Input tensor
        input_feats = Input(shape=self.input_shape, name='input_layer')

        conv_layer = Conv1D(128, kernel_size=3, padding='same', name='conv1d')(input_feats)
        max_pooling = MaxPooling1D(3,1,padding='same', name='pool1')(conv_layer)

        gru = GRU(128, return_sequences=True, name='gru')(max_pooling)

        self.model = Model(input_feats, gru)
        self.predictive_m = Model(max_pooling, gru)

    def load_prediction_configuration(self, config):
        pass

    def train(self):
        pass

    def predict(self, x_test, x_test_ind, duration):
        pass