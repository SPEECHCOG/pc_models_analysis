"""
@date 16.04.2020
Contrastive Predictive Coding model
[Representation Learning with Contrastive Predictive Coding]
"""
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from keras import backend as K
from sklearn.decomposition import PCA
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Dropout, Conv1D, Input, GRU
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

from models.convpc import Block, ContrastiveLoss
from models.create_prediction_files import create_prediction_files
from models.model_base import ModelBase

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


class FeatureEncoder(Block):
    """
    It creates a keras layer for the encoder part (latent representations)
    """
    def __init__(self, n_layers, units, dropout, name='Feature_Encoder'):
        """
        :param n_layers: Number of convolutional layers
        :param units: Number of filters per convolutional layer
        :param dropout: Percentage of dropout between layers
        :param name: Name of the block, by default Feature_Encoder
        """
        super(FeatureEncoder, self).__init__(name=name)
        self.n_layers = n_layers
        self.units = units
        self.dropout = dropout
        self.layers = []
        with K.name_scope(name):
            for i in range(n_layers):
                self.layers.append(Conv1D(units, 3, padding='causal', activation='relu', name='conv_layer_'+str(i)))
                if i == n_layers - 1:
                    self.layers.append(Dropout(dropout, name='cpc_latent_layer'))
                else:
                    self.layers.append(Dropout(dropout, name='conv_dropout_' + str(i)))

    def call(self, inputs, **kwargs):
        """
        It is execute when an input tensor is passed
        :param inputs: A tensor with the input features
        :return: A tensor with the output of the block
        """
        features = inputs
        for layer in self.layers:
            features = layer(features)
        return features

    def get_config(self):
        return {'n_layers': self.n_layers, 'units': self.units, 'dropout': self.dropout}


class CPCModel(ModelBase):

    def write_log(self, file_path):
        """
        Write the log of the configuration parameters used for training a CPC model
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
            f.write('CPC configuration: \n')
            for param in self.configuration['model']['cpc']:
                f.write(param + ': ' + str(self.configuration['model']['cpc'][param]) + '\n')

    def load_prediction_configuration(self, config):
        """
        It uses implementation from ModelBase
        :param config: dictionary with the configuration for predictions
        :return: the instance model have all configuration parameters from config
        """
        super(CPCModel, self).load_prediction_configuration(config)
        self.use_pca = config['use_pca']

    def load_training_configuration(self, config, x_train, y_train):
        """
        It instantiates the model architecture using the parameters from the configuration file.
        :param config: Dictionary with the configuration parameters
        :param x_train: Input training data
        :param y_train: Output training data (not used in this model)
        :return: an instance will have the parameters from configuration and the model architecture
        """
        super(CPCModel, self).load_training_configuration(config, x_train, y_train)

        # Model architecture: Feature_Encoder -> Dropout -> GRU -> Dropout
        cpc_config = config['model']['cpc']
        # feature encoder params
        self.encoder_layers = cpc_config['encoder_layers']
        self.encoder_units = cpc_config['encoder_units']
        self.encoder_dropout = cpc_config['encoder_dropout']

        # autoregressive model params
        self.gru_units = cpc_config['gru_units']

        # contrastive loss params
        self.neg = cpc_config['neg']
        self.steps = cpc_config['steps']

        # dropout and learning rate params
        self.dropout = cpc_config['dropout']
        self.learning_rate = cpc_config['learning_rate']

        # Input shape
        self.input_shape = self.x_train.shape[1:]
        self.features = self.x_train.shape[2]

        # Input tensor
        input_feats = Input(shape=self.input_shape, name='input_layer')

        # Dropout layer
        dropout_layer = Dropout(self.dropout, name='dropout_block')

        # Feature Encoder
        feature_encoder = FeatureEncoder(self.encoder_layers, self.encoder_units, self.encoder_dropout)
        encoder_features = feature_encoder(input_feats)
        encoder_output = dropout_layer(encoder_features)

        # Autoregressive model
        autoregressive_model = GRU(self.gru_units, return_sequences=True, name='autoregressive_layer')
        autoregressive_output = autoregressive_model(encoder_output)
        autoregressive_output = dropout_layer(autoregressive_output)

        # Contrastive loss
        contrastive_loss = ContrastiveLoss(self.gru_units, self.neg, self.steps)
        contrastive_loss_output = contrastive_loss([encoder_features, autoregressive_output])

        # Model
        self.model = Model(input_feats, contrastive_loss_output)

        print(self.model.summary())

    def train(self):
        """
        Train a CPC model
        :return: a trained model saved on disk
        """
        adam = Adam(lr=self.learning_rate)
        self.model.compile(optimizer=adam, loss={'Contrastive_Loss': lambda y_true, y_pred: y_pred})

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
        log_dir = os.path.join(self.logs_folder_path, self.language, datetime.now().strftime("%Y_%m_%d-%H_%M"))
        tensorboard = TensorBoard(log_dir=log_dir, write_graph=True, profile_batch=0)

        # Train the model
        # Create dummy prediction so that Keras does not raise an error for wrong dimension
        y_dummy = np.random.rand(self.x_train.shape[0], 1, 1)

        self.model.fit(x=self.x_train, y=y_dummy, epochs=self.epochs, batch_size=self.batch_size,
                       validation_split=0.3, callbacks=[tensorboard, early_stop, checkpoint])

        return self.model

    def predict(self, x_test, x_test_ind, duration):
        """
        It predicts the representation for input test set.
        :param x_test: a numpy array with the test features
        :param x_test_ind: a numpy array as a lookup table for matching frames with utterances
        :param duration: duration of the utterances
        :return: predictions for the test set in txt files
        """
        self.model = load_model(self.model_path, compile=False, custom_objects={'FeatureEncoder': FeatureEncoder,
                                                                                'ContrastiveLoss': ContrastiveLoss})

        if self.use_last_layer:
            # Predict using the latent representations (APC output)
            input_layer = self.model.get_layer('input_layer').output
            latent_layer = self.model.get_layer('Feature_Encoder').get_layer('cpc_latent_layer').output
            predictor = Model(input_layer, latent_layer)

            predictions = predictor.predict(x_test)
        else:
            # Predict using the context representations (CPC output)
            input_layer = self.model.get_layer('input_layer').output
            context_layer = self.model.get_layer('autoregressive_layer').output
            predictor = Model(input_layer, context_layer)

            predictions = predictor.predict(x_test)

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

