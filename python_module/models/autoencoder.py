"""
@date 25.02.2020

Autoencoder model
"""
import os
from datetime import datetime

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import Model
from keras.layers import Input, Dense

from models.model_base import ModelBase


class Autoencoder(ModelBase):
    def load_training_configuration(self, config, x_train, y_train):
        """
        The autoencoder has the following architecture:
        * 6 layers in total
        * 3 layers for encoder and 3 layers for decoder
        * Input -> Encoder (stacked Dense layers) -> Decoder (stacked Dense layers) -> Output

        :param config: a dictionary with the configuration parameters for training
        :param x_train: a numpy 3D-array with the samples (input features) to train the model
        :param y_train: a numpy 3D-array with the samples (output features) to train the model
        """
        super(Autoencoder, self).load_training_configuration(config, x_train, y_train)

        self.input_shape = self.x_train.shape[1:]
        self.features = self.x_train.shape[2]

        # Model architecture
        # Input layer
        input_feats = Input(shape=self.input_shape, name='input_layer')

        # Stacked encoder
        encoded_1 = Dense(128, activation='relu', name='encoded_1')(input_feats)
        encoded_2 = Dense(64, activation='relu', name='encoded_2')(encoded_1)
        encoded_3 = Dense(self.latent_dimension, activation='relu', name='latent_layer')(encoded_2)

        # Stacked decoder
        decoded_1 = Dense(64, activation='relu', name='decoded_1')(encoded_3)
        decoded_2 = Dense(128, activation='relu', name='decoded_2')(decoded_1)
        decoded_3 = Dense(self.features, activation='linear', name='decoded_3')(decoded_2)

        self.model = Model(input_feats, decoded_3)

    def load_prediction_configuration(self, config):
        """
        It loads the configuration for predictions using default behaviour from ModelBase
        :param config: a dictionary with the configuration parameters
        :return: the instance will have the configuration parameters
        """
        super(Autoencoder, self).load_prediction_configuration(config)

    def train(self):
        """
        It trains an autoencoder model using the parameters in the configuration.
        :return: a trained model. The model saved in h5 file
        """

        # Configuration of learning process
        self.model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])

        # Callbacks for training
        # Adding early stop based on validation loss and saving best model for later prediction

        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=self.early_stop_epochs)
        checkpoint = ModelCheckpoint(os.path.join(self.full_path_output_folder, self.language +
                                                  datetime.now().strftime("_%Y_%m_%d-%H_%M") + '.h5'),
                                     monitor='val_mean_absolute_error', mode='min', verbose=1, save_best_only=True)

        # Tensorboard
        log_dir = os.path.join(self.logs_folder_path, datetime.now().strftime("%Y_%m_%d-%H_%M"))
        tensorboard = TensorBoard(log_dir=log_dir, write_graph=True)

        # Train the model
        self.model.fit(self.x_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.3,
                       callbacks=[tensorboard, early_stop, checkpoint])

        return self.model

    def predict(self, x_test, x_test_ind, duration):
        """
        It uses default behaviour from ModelBase
        """
        super(Autoencoder, self).predict(x_test, x_test_ind, duration)
