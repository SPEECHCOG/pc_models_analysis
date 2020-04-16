""""
@date 26.02.2020
Autoregressive Predictive Coding model
[An unsupervised autoregressive model for speech representation learning]

This corresponds to a translation from the Pytorch implementation
(https://github.com/iamyuanchung/Autoregressive-Predictive-Coding) to Keras implementation
"""
import os
from datetime import datetime

from sklearn.decomposition import PCA
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Input, Dense, Dropout, GRU, Add, Conv1D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

from models.create_prediction_files import create_prediction_files
from models.model_base import ModelBase

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


class APCModel(ModelBase):

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
            f.write('latent dimension: ' + str(self.latent_dimension) + '\n\n')
            f.write('APC configuration: \n')
            for param in self.configuration['model']['apc']:
                if param.startswith('prenet_'):
                    if self.prenet:
                        f.write(param + ': ' + str(self.configuration['model']['apc'][param]) + '\n')
                else:
                    f.write(param + ': ' + str(self.configuration['model']['apc'][param]) + '\n')

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
        adam = Adam(lr=self.learning_rate)
        self.model.compile(optimizer=adam, loss='mean_absolute_error')

        # Model file name for checkpoint and log
        model_file_name = os.path.join(self.full_path_output_folder, self.language +
                                       datetime.now().strftime("_%Y_%m_%d-%H_%M"))

        # log
        self.write_log(model_file_name + '.txt')

        # Callbacks for training
        # Adding early stop based on validation loss and saving best model for later prediction
        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=self.early_stop_epochs)
        checkpoint = ModelCheckpoint( model_file_name + '.h5', monitor='val_loss', mode='min',
                                      verbose=1, save_best_only=True)

        # Tensorboard
        log_dir = os.path.join(self.logs_folder_path, self.language, datetime.now().strftime("%Y_%m_%d-%H_%M"))
        tensorboard = TensorBoard(log_dir=log_dir, write_graph=True, profile_batch=0)

        # Train the model
        self.model.fit(self.x_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.3,
                       callbacks=[tensorboard, early_stop, checkpoint])

        return self.model

    def predict(self, x_test, x_test_ind, duration):
        """
        It predicts the output features for the test set (x_test) and output the predictions in text files using
        (x_test_ind). Instead of directly using the full size of latent representation, we used PCA analysis to reduce
        dimensionality.
        :param x_test: a numpy array with the test set (samples of input features in the same format than those used
                       for training the model). It has dimension samples x time-steps x features.
        :param x_test_ind: a numpy array with the indices (number of frame in the source audio) for each sample. The
                           dimension is samples x time-steps x 2 (where the first number is the source audio identifier,
                           and the second one is the number of the frame in the audio).
        :param duration: a string with the duration of the audio files (1, 10 or 120)
        :return: predictions will be saved in text files. The folder structure is kept.
        """
        self.model = load_model(self.model_path)

        if self.use_last_layer:
            predictor = self.model
        else:
            # Prediction of model will use latent representation (intermediate layer)
            input_layer = self.model.get_layer('input_layer').output
            latent_layer = self.model.get_layer('latent_layer').output
            predictor = Model(input_layer, latent_layer)

        # Calculate predictions dimensions (samples, 200, latent-dimension)
        predictions = predictor.predict(x_test)

        # Apply PCA only for latent layer representations
        if not self.use_last_layer:
            pca = PCA(0.95) # Keep components that coverage 95% of variance
            pred_orig_shape = predictions.shape
            predictions = predictions.reshape(-1, predictions.shape[-1])
            predictions = pca.fit_transform(predictions)
            pred_orig_shape = list(pred_orig_shape)
            pred_orig_shape[-1] = predictions.shape[-1]
            pred_orig_shape = tuple(pred_orig_shape)
            predictions = predictions.reshape(pred_orig_shape)

        # Create folder for predictions
        full_predictions_folder_path = os.path.join(self.output_folder,self.model_folder_name,
                                                    self.features_folder_name, self.language, (duration + 's'))
        os.makedirs(full_predictions_folder_path, exist_ok=True)

        if self.save_matlab:
            # Save predictions in MatLab file using h5py formatting
            import hdf5storage
            output = dict()
            output['pred'] = predictions
            output['pred_ind'] = x_test_ind
            hdf5storage.savemat(os.path.join(full_predictions_folder_path, self.language + '.mat'), output,
                                format='7.3')

        # Create predictions text files
        total_files = create_prediction_files(predictions, x_test_ind, full_predictions_folder_path, self.window_shift,
                                              limit=self.files_limit)

        print('Predictions of {0} with duration {1}s: {2} files'.format(self.language, duration, total_files))

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
        # APC (multi-layer GRU network) -> Postnet(Conv1D this is only used during training)

        # Load apc configuration parameters
        apc_config = config['model']['apc']
        self.prenet = apc_config['prenet']
        self.prenet_layers = apc_config['prenet_layers']
        self.prenet_dropout = apc_config['prenet_dropout']
        self.prenet_units = apc_config['prenet_units']
        self.rnn_layers = apc_config['rnn_layers']
        self.rnn_dropout = apc_config['rnn_dropout']
        self.rnn_units = apc_config['rnn_units']
        self.residual = apc_config['residual']
        self.learning_rate = apc_config['learning_rate']

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
