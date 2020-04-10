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
from tensorflow.keras.layers import Input, Conv1D, Layer, Lambda, Dense, Dropout, Add, Conv2DTranspose
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

from models.create_prediction_files import create_prediction_files
from models.model_base import ModelBase

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# Blocks used in the ConvPC architecture:
# * Encoder (Prenet)
# * APC (Convolutions)
# * CPC (Convolutions <context part>)
# * ContrastiveLoss (Contrastive loss calculation)
class Block(Layer):
    """
    Super class for all the blocks so they have get_layer method. The method is used in prediction to extract either
    features of the APC encoder or the CPC encoder
    """
    def __init__(self, name):
        super(Block, self).__init__(name=name)

    def get_layer(self, name=None, index=None):
        """
        Keras sourcecode for Model.
        :param name: String name of the layer
        :param index: int index of the layer
        :return: the layer if name or index is found, error otherwise
        """
        if index is not None:
            if len(self.layers) <= index:
                raise ValueError('Was asked to retrieve layer at index ' + str(index) +
                                 ' but model only has ' + str(len(self.layers)) +
                                 ' layers.')
            else:
                return self.layers[index]
        else:
            if not name:
                raise ValueError('Provide either a layer name or layer index.')
            for layer in self.layers:
                if layer.name == name:
                    return layer
            raise ValueError('No such layer: ' + name)


class Prenet(Block):
    """
    It creates a keras layer for the encoder part
    """
    def __init__(self, n_layers, units, dropout, name='PreNet'):
        """
        :param n_layers: Number of the full connected layers
        :param units: Number of units per layer
        :param dropout: Percentage of dropout between layers
        :param name: Name of the block, by default PreNet as in APC
        """
        super(Prenet, self).__init__(name=name)
        self.layers = []
        self.n_layers = n_layers
        self.units = units
        self.dropout = dropout
        with K.name_scope(name):
            for i in range(n_layers):
                self.layers.append(Dense(units, activation='relu', name='prenet_linear_' + str(i)))
                self.layers.append(Dropout(dropout, name='prenet_dropout_' + str(i)))

    def call(self, inputs, **kwargs):
        """
        It is execute when an input tensor is passed
        :param inputs: A tensor with the input features
        :return: A tensor with the output of the block
        """
        features = inputs
        for i in range(len(self.layers)):
            features = self.layers[i](features)
        return features

    def get_config(self):
        return {'n_layers': self.n_layers, 'units': self.units, 'dropout': self.dropout}


class APC(Block):
    """
    It creates a keras layer for the APC part
    :return: A tensor with the output of the block
    """
    def __init__(self, n_layers, units, residual, dropout, name='APC'):
        """
        :param n_layers: Number of convolutional layers
        :param units: Number of filters per convolutional layer
        :param residual: A boolean stating if residual connections are used or not
        :param dropout: Percentage of dropout between layers
        :param name: Name of the block, by default APC

        """
        super(APC, self).__init__(name=name)
        self.layers = []
        self.n_layers = n_layers
        self.layers_type = []
        self.residual = residual
        self.units = units
        self.dropout = dropout

        with K.name_scope(name):
            for i in range(n_layers):
                if not residual and i+1 == n_layers:
                    self.layers.append(Conv1D(units, 3, padding='causal', activation='relu',
                                              name='apc_latent_layer'))
                    self.layers_type.append('conv')
                else:
                    self.layers.append(Conv1D(units, 3, padding='causal', activation='relu',
                                              name='conv_layer_' + str(i)))
                    self.layers_type.append('conv')

                if i+1 < n_layers:
                    # Dropout to all layers except last layer
                    self.layers.append(Dropout(dropout, name='conv_dropout_'+str(i)))
                    self.layers_type.append('dropout')

                if residual:
                    if i+1 < n_layers:
                        self.layers.append(Add(name='conv_residual_'+str(i)))
                        self.layers_type.append('add')

                    # Update output for next layer (PostNet) if this is the last layer. This will also be the latent
                    # representation.
                    if i+1 == n_layers:
                        self.layers.append(Add(name='apc_latent_layer'))
                        self.layers_type.append('add')

    def call(self, inputs, **kwargs):
        """
        It is execute when an input tensor is passed
        :param inputs: A tensor with the input features
        :return: A tensor with the output of the block
        """
        n_feats = inputs.shape[-1]
        conv_input = inputs

        residual_first = False
        if self.residual:
            if self.units != n_feats:
                # The first residual connection cannot be made, therefore the Add layer can be removed
                if self.layers_type[2] == 'add':
                    del self.layers_type[2]
                    del self.layers[2]
                residual_first = False
            else:
                residual_first = True

        conv_output = None
        for i, layer in enumerate(self.layers):
            if self.layers_type[i] == 'conv':
                conv_output = layer(conv_input)
            if self.layers_type[i] == 'dropout':
                if not self.residual or (not residual_first and i == 1):
                    # When there is not residual connection after the dropout, this is the input for the next
                    # convolution
                    conv_input = layer(conv_output)
                else:
                    conv_output = layer(conv_output)
            if self.layers_type[i] == 'add':
                conv_input = layer([conv_input, conv_output])
                if i+1 == len(self.layers):
                    # If it is the last layer, then this is the output of the block
                    conv_output = conv_input
        return conv_output

    def get_config(self):
        return {'n_layers': self.n_layers, 'units': self.units, 'dropout': self.dropout, 'residual': self.residual}


class CPC(Block):
    """
    It creates a keras layer for the CPC part
    """
    def __init__(self, n_layers, units, dropout, name='CPC'):
        """
        :param n_layers: Number of convolutional layers
        :param units: Number of filters per convolutional layer
        :param dropout: Percentage of dropout between layers
        :param name: Name of the block, by default CPC
        """
        super(CPC, self).__init__(name=name)
        self.n_layers = n_layers
        self.units = units
        self.dropout = dropout
        self.layers = []
        with K.name_scope(name):
            for i in range(n_layers):
                self.layers.append(Conv1D(units, 3, padding='causal', activation='relu', name='conv_layer_'+str(i)))
                if i == n_layers - 1:
                    self.layers.append(Dropout(dropout, name='cpc_context'))
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


class ContrastiveLoss(Block):
    """
    It creates the block that calculates the contrastive loss for given latent representation and context
    representations. Implementation from wav2vec
    (https://github.com/pytorch/fairseq/blob/master/fairseq/models/wav2vec.py)
    [wav2vec: Unsupervised Pre-training for Speech Recognition](https://arxiv.org/abs/1904.05862)
    """
    def __init__(self, context_units, neg, steps, name='Contrastive_Loss'):
        """
        :param context_units: Number of units of the context representation
        :param neg: Number of negative samples
        :param steps: Number of steps to predict
        :param name: Name of the block, by default Contrastive_Loss
        """
        super(ContrastiveLoss, self).__init__(name=name)
        self.neg = neg
        self.steps = steps
        self.context_units = context_units
        self.layers = []
        with K.name_scope(name):
            self.project_steps = Conv2DTranspose(self.steps, kernel_size=1, strides=1, name='project_steps')
            self.project_latent = Conv1D(self.context_units, kernel_size=1, strides=1, name='project_latent')
            self.cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=True,
                                                                         reduction=tf.keras.losses.Reduction.SUM)
            self.layers.append(self.project_steps)
            self.layers.append(self.project_latent)

    def get_negative_samples(self, true_features):
        """
        It calculates the negative samples re-ordering the time-steps of the true features.
        :param true_features: A tensor with the apc predictions for the input.
        :return: A tensor with the negative samples.
        """
        # Shape SxTxF
        samples = K.shape(true_features)[0]
        timesteps = K.shape(true_features)[1]
        features = K.shape(true_features)[2]

        # New shape FxSxT
        true_features = K.permute_dimensions(true_features, pattern=(2, 0, 1))
        # New shape Fx (S*T)
        true_features = K.reshape(true_features, (features, -1))

        high = timesteps

        # New order for time-steps
        indices = tf.repeat(tf.expand_dims(tf.range(timesteps), axis=-1), self.neg)
        neg_indices = tf.random.uniform(shape=(samples, self.neg * timesteps), minval=0, maxval=high - 1,
                                        dtype=tf.dtypes.int32)
        neg_indices = tf.where(tf.greater_equal(neg_indices, indices), neg_indices + 1, neg_indices)

        right_indices = tf.reshape(tf.range(samples), (-1, 1))*high
        neg_indices = neg_indices + right_indices

        # Reorder for negative samples
        negative_samples = tf.gather(true_features, tf.reshape(neg_indices, [-1]), axis=1)
        negative_samples = K.permute_dimensions(K.reshape(negative_samples,
                                                          (features, samples, self.neg, timesteps)),
                                                (2, 1, 3, 0))
        return negative_samples

    def call(self, inputs, **kwargs):
        """
        :param inputs: A list with two elements, the latent representation and the context representation
        :param kwargs:
        :return: the contrastive loss calculated
        """
        true_latent, context_latent = inputs

        # Linear transformation of latent representation into the vector space of context representations
        true_latent = self.project_latent(true_latent)

        # Calculate the following steps using context_latent
        context_latent = K.expand_dims(context_latent, -1)
        predictions = self.project_steps(context_latent)

        negative_samples = self.get_negative_samples(true_latent)

        true_latent = K.expand_dims(true_latent, 0)

        targets = K.concatenate([true_latent, negative_samples], 0)
        copies = self.neg + 1  # total of samples in targets

        # samples, timesteps, features, steps = predictions.shape

        # Logits calculated from predictions and targets
        logits = None

        for i in range(self.steps):
            if i == 0:
                # The time-steps are correspondent as is the first step.
                logits = tf.reshape(tf.einsum("stf,cstf->tsc", predictions[:, :, :, i], targets[:, :, :, :]), [-1])
            else:
                # We need to match the time-step taking into account the step for which is being predicted
                logits = tf.concat([logits, tf.reshape(tf.einsum("stf,cstf->tsc", predictions[:, :-i, :, i],
                                                                 targets[:, :, i:, :]), [-1])], 0)

        logits = tf.reshape(logits, (-1, copies))
        total_points = tf.shape(logits)[0]

        # Labels, this should be the true value, that is 1.0 for the first copy (positive sample) and 0.0 for the
        # rest.
        label_idx = [True] + [False] * self.neg
        labels = tf.where(label_idx, tf.ones((total_points, copies)), tf.zeros((total_points, copies)))

        # The loss is the softmax_cross_entropy_with_logits sum over all steps.
        loss = self.cross_entropy(labels, logits)
        loss = tf.reshape(loss, (1,))
        return loss

    def get_config(self):
        return {'context_units': self.context_units, 'neg': self.neg, 'steps': self.steps}


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
        # Prenet params
        self.prenet = convpc_config['prenet']
        self.prenet_layers = convpc_config['prenet_layers']
        self.prenet_dropout = convpc_config['prenet_dropout']
        self.prenet_units = convpc_config['prenet_units']

        # APC params
        self.apc_residual = convpc_config['apc_residual']
        self.apc_layers = convpc_config['apc_layers']
        self.apc_dropout = convpc_config['apc_dropout']
        self.apc_units = convpc_config['apc_units']

        # CPC params
        self.cpc_layers = convpc_config['cpc_layers']
        self.cpc_dropout = convpc_config['cpc_dropout']
        self.cpc_units = convpc_config['cpc_units']
        self.cpc_neg = convpc_config['cpc_neg']
        self.cpc_steps = convpc_config['cpc_steps']

        self.dropout = convpc_config["dropout"]
        self.learning_rate = convpc_config['learning_rate']

        # Define model
        # input size and number of features. Input is numpy array of size (samples, time-steps, features)
        self.input_shape = self.x_train.shape[1:]
        self.features = self.x_train.shape[2]

        # Dropout layer
        dropout_layer = Dropout(self.dropout, name='dropout_blocks')
        # Input tensor
        input_feats = Input(shape=self.input_shape, name='input_layer')

        # Encoding part
        prenet_block = Prenet(self.prenet_layers, self.prenet_units, self.prenet_dropout)
        prenet_output = prenet_block(input_feats)

        # APC Part
        apc_block = APC(self.apc_layers, self.apc_units, self.apc_residual, self.apc_dropout)
        apc_features = apc_block(prenet_output)
        # Dropout before passing to CPC
        apc_output = dropout_layer(apc_features)

        # CPC Part
        cpc_block = CPC(self.cpc_layers, self.cpc_units, self.cpc_dropout)
        cpc_output = cpc_block(apc_output)
        cpc_output = dropout_layer(cpc_output)

        # Use for restricting latent representations
        autoencoder = Conv1D(self.features, kernel_size=1, strides=1, padding='same', name='apc_posnet')
        autoencoder_prediction = autoencoder(apc_output)

        contrastive_loss = ContrastiveLoss(self.cpc_units, self.cpc_neg, self.cpc_steps)
        contrastive_loss_output = contrastive_loss([apc_output, cpc_output])

        # Model
        self.model = Model(input_feats, [contrastive_loss_output, autoencoder_prediction])

        print(self.model.summary())

    def load_prediction_configuration(self, config):
        """
                It uses implementation from ModelBase
                :param config: dictionary with the configuration for predictions
                :return: the instance model have all configuration parameters from config
                """
        super(ConvPCModel, self).load_prediction_configuration(config)

        self.use_pca = config['use_pca']

    def train(self):
        """
        Training ConvPC will require two losses, one optimising the autoencoder and the second optimising the gru
        prediction.
        :return: a trained model saved on disk
        """

        # Configuration of learning process
        adam = Adam(lr=self.learning_rate)
        self.model.compile(optimizer=adam, loss={'Contrastive_Loss': lambda y_true, y_pred: y_pred,
                                                 'apc_posnet': 'mean_absolute_error'})

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

        # Use + N (10) frames in future to train future latent representations.
        y_future = np.roll(self.y_train.reshape(self.y_train.shape[0]*self.y_train.shape[1], self.y_train.shape[-1]),
                           -15, axis=0).reshape(self.y_train.shape)

        self.model.fit(x=self.x_train, y=[y_dummy, self.y_train], epochs=self.epochs, batch_size=self.batch_size,
                       validation_split=0.3, callbacks=[tensorboard, early_stop, checkpoint])

        return self.model

    def predict(self, x_test, x_test_ind, duration):

        self.model = load_model(self.model_path, compile=False, custom_objects={'Prenet': Prenet, 'APC': APC,
                                                                                'CPC': CPC,
                                                                                'ContrastiveLoss': ContrastiveLoss})

        if self.use_last_layer:
            # Predict using the latent representations (APC output)
            input_layer = self.model.get_layer('input_layer').output
            latent_layer = self.model.get_layer('APC').get_layer('apc_latent_layer').output
            predictor = Model(input_layer, latent_layer)

            predictions = predictor.predict(x_test)
        else:
            # Predict using the context representations (CPC output)
            input_layer = self.model.get_layer('input_layer').output
            context_layer = self.model.get_layer('CPC').get_layer('cpc_context').output
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
            import scipy.io
            scipy.io.savemat(os.path.join(full_predictions_folder_path, self.language + '.mat'),
                             {'pred': predictions, 'pred_ind': x_test_ind})

        # Create predictions text files
        total_files = create_prediction_files(predictions, x_test_ind, full_predictions_folder_path, self.window_shift,
                                              limit=self.files_limit)

        print('Predictions of {0} with duration {1}s: {2} files'.format(self.language, duration, total_files))
