"""
    @author Maria Andrea Cruz Blandon
    @date 05.03.2020

    Convolutional Predictive Coding:  In this model we use the schematic idea of Contrastive Predictive Coding in which
    we learn short-term representation and context-term representations, and ideas of Autoregressive Predictive Coding
    for calculate predictions of context-term representations.
"""
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from keras import backend as K
from sklearn.decomposition import PCA
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Concatenate, Lambda
from tensorflow.keras.losses import mean_absolute_error
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

from models.create_prediction_files import create_prediction_files
from models.model_base import ModelBase


class ConvPCModel(ModelBase):

    def write_log(self, file_path):
        """
        Write the log of the configuration parameters used for training an APC model
        :param file_path: path where the file will be saved
        :return: a text logfile
        """
        with open(file_path, 'w+', encoding='utf8') as f:
            f.write('Training configuration: \n')
            f.write('Source: ' + self.configuration['train_in'][0] + '\n')
            f.write('Target: ' + self.configuration['train_out'][0] + '\n')
            f.write('Features: ' + self.features_folder_name + '\n')
            f.write('Language: ' + self.language + '\n\n')
            f.write('Model configuration: \n')
            f.write('epochs: ' + str(self.epochs) + '\n')
            f.write('early stop epochs: ' + str(self.early_stop_epochs) + '\n')
            f.write('batch size: ' + str(self.batch_size) + '\n')
            f.write('ConvPC configuration: \n')
            for param in self.configuration['model']['convpc']:
                f.write(param + ': ' + str(self.configuration['model']['convpc'][param]) + '\n')

    def load_training_configuration(self, config, x_train, y_train):
        """
        Load training configuration from configuration dictionary. And create model architecture. This model have an
        encoder for short-term analysis and then a predictive coding part using latent representations of the encoder
        for prediction.
        :param config: a dictionary with the configuration parameters for training
        :param x_train: a numpy array with the input features
        :param y_train: a numpy array with the output features
        :return: an instance will have the parameters from configuration and the model architecture
        """
        super(ConvPCModel, self).load_training_configuration(config, x_train, y_train)

        # Load ConvPC configurations
        convpc_config = config['model']['convpc']
        self.conv_units = convpc_config['conv_units']
        self.learning_rate = convpc_config['learning_rate']

        # Define model
        # input size and number of features. Input is numpy array of size (samples, time-steps, features)
        self.input_shape = self.x_train.shape[1:]
        self.features = self.x_train.shape[2]

        # Input tensor
        input_feats = Input(shape=self.input_shape, name='input_layer')
        input_feats_future = Input(shape=self.input_shape, name='input_future_layer')

        # Encoding part
        units = [self.conv_units, 16, 8, 16, self.conv_units]
        conv_layers = []
        maxpool_layers = []

        for i, unit in enumerate(units):
            conv_layers.append(Conv1D(units[i], kernel_size=3, padding='causal', activation='relu',
                                      name='conv_' + str(i)))
            if i == len(units) - 1:
                maxpool_layers.append(MaxPooling1D(3, 1, padding='same', name='latent_layer'))
            else:
                maxpool_layers.append(MaxPooling1D(3, 1, padding='same', name='pool_' + str(i)))

        # Use for restricting latent representations
        autoencoder = Conv1D(self.features, kernel_size=1, strides=1, padding='same', name='autoencoder')

        # Predictive coding part
        latent_layers = []
        maxpool_latent_layers = []
        future_layer = Conv1D(self.conv_units, kernel_size=5, padding='causal', activation='relu', name='gru')

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

        future_prediction = future_layer(future_prediction)
        autoencoder_prediction = autoencoder(autoencoder_prediction)

        # concatenate gru predictions and latent_future (y_true)
        model_out = Concatenate()([future_prediction, latent_future])

        def final_loss_func(args):
            """
            It calculates the absolute difference between the autoencoder MAE and future representations MAE.
            :param args: list containing the y_true for autoencoder, y_pred for autoencoder, y_true of latent
                         representations and y_pred of latent representations
            :return: tensor with the absolute value of the difference
            """
            auto_true, auto_pred, latent_true, latent_pred = args
            auto_mae = K.mean(K.abs((auto_pred - auto_true)), keepdims=True)
            latent_mae = K.mean(K.abs(latent_pred - latent_true), keepdims=True)
            return K.abs(auto_mae - latent_mae)

        # Minimisation of difference between MAE of autoencoder and MAE of future latent representations
        final_loss = Lambda(final_loss_func, output_shape=(1,), name='final_loss')([input_feats, autoencoder_prediction,
                                                                                    latent_future, future_prediction])

        # Model
        self.model = Model([input_feats, input_feats_future], [final_loss, model_out, autoencoder_prediction])

        # print(self.model.summary())

        gpus = tf.config.experimental.list_physical_devices('GPU')

        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

    def load_prediction_configuration(self, config):
        """
                It uses implementation from ModelBase
                :param config: dictionary with the configuration for predictions
                :return: the instance model have all configuration parameters from config
                """
        super(ConvPCModel, self).load_prediction_configuration(config)

        self.use_pca = config['use_pca']
        self.conv_units = config['convpc']['conv_units']

    def train(self):
        """
        Training ConvPC will require two losses, one optimising the autoencoder and the second optimising the gru
        prediction.
        :return: a trained model saved on disk
        """

        def mae_latent(y_true, y_pred):
            """
            Calcuate mean absolute error for GRU predictions in the latent space
            :param y_true: tensor with true values (it is not used, as it is a random matrix)
            :param y_pred: tensor with predictions (concatenation of latent representation of future frames and gru
                           predictions
            :return: mean absolute error
            """
            mae = mean_absolute_error(y_pred[:, :, :self.conv_units], y_pred[:, :, self.conv_units:])
            return mae

        # Configuration of learning process
        adam = Adam(lr=self.learning_rate)
        self.model.compile(optimizer=adam, loss=[lambda y_true, y_pred: y_pred, mae_latent,
                                                 'mean_absolute_error'], loss_weights=[0.5, 0.1, 0.4])

        # Model file name for checkpoint and log
        model_file_name = os.path.join(self.full_path_output_folder, self.language +
                                       datetime.now().strftime("_%Y_%m_%d-%H_%M"))

        # log
        self.write_log(model_file_name + '.txt')

        # Callbacks for training
        # Adding early stop based on validation loss and saving best model for later prediction
        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=self.early_stop_epochs)
        checkpoint = ModelCheckpoint(model_file_name + '.h5', monitor='val_loss', mode='min',
                                     verbose=1, save_best_only=True)

        # Tensorboard
        log_dir = os.path.join(self.logs_folder_path, datetime.now().strftime("%Y_%m_%d-%H_%M"))
        tensorboard = TensorBoard(log_dir=log_dir, write_graph=True, profile_batch=0)

        # Train the model
        # Create dummy prediction so that Keras does not raise an error for wrong dimension
        y_dummy = np.random.rand(self.x_train.shape[0], 1, 1)

        self.model.fit([self.x_train, self.y_train], [y_dummy, y_dummy, self.x_train], epochs=self.epochs,
                       batch_size=self.batch_size, validation_split=0.3,
                       callbacks=[tensorboard, early_stop, checkpoint])

        return self.model

    def predict(self, x_test, x_test_ind, duration):

        self.model = load_model(self.model_path, compile=False)

        if self.use_last_layer:
            predictor = self.model
            # Calculate predictions dimensions (samples, 200, latent-dimension)
            _, predictions, _ = predictor.predict([x_test, x_test])
            # only first part of the concatenated output
            predictions = predictions[:, :, :self.conv_units]
        else:
            # Prediction of model will use latent representation (intermediate layer)
            input_layer = self.model.get_layer('input_layer').output
            input_future_layer = self.model.get_layer('input_future_layer').output
            latent_layer = self.model.get_layer('latent_layer').output
            predictor = Model([input_layer, input_future_layer], latent_layer)

            # Calculate predictions dimensions (samples, 200, latent-dimension)
            predictions = predictor.predict([x_test, x_test])

        # Apply PCA only if true in the configuration file.
        if self.use_pca:
            pca = PCA(0.95)  # Keep components that coverage 95% of variance
            pred_orig_shape = predictions.shape
            predictions = predictions.reshape(-1, predictions.shape[-1])
            predictions = pca.fit_transform(predictions)
            pred_orig_shape = list(pred_orig_shape)
            pred_orig_shape[-1] = predictions.shape[-1]
            pred_orig_shape = tuple(pred_orig_shape)
            predictions = predictions.reshape(pred_orig_shape)

        # Create folder for predictions
        full_predictions_folder_path = os.path.join(self.output_folder, self.model_folder_name,
                                                    self.features_folder_name, self.language, (duration + 's'))
        os.makedirs(full_predictions_folder_path, exist_ok=True)

        # Create predictions text files
        total_files = create_prediction_files(predictions, x_test_ind, full_predictions_folder_path, self.window_shift,
                                              limit=self.files_limit)

        print('Predictions of {0} with duration {1}s: {2} files'.format(self.language, duration, total_files))
