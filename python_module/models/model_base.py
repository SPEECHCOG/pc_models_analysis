"""
@date 25.02.2020
Abstract class for neural networks models using Keras.
New models should inherit from this class so we can run training and prediction the same way for
all the models and independently.
"""
import os
from abc import ABC, abstractmethod

from keras.models import Model
from keras.engine.saving import load_model

from models.create_prediction_files import create_prediction_files


class ModelBase(ABC):
    @abstractmethod
    def load_training_configuration(self, config, x_train, y_train):
        """
        Loads the configuration parameters from the configuration dictionary, and the input/output features for
        training
        :param config: a dictionary with the configuration for training
        :param x_train: a numpy array with the input features
        :param y_train: a numpy array with the output features
        :return: instance will have the configuration parameters
        """
        # properties from configuration file for training
        model_config = config['model']
        self.name = model_config['name']
        self.type = model_config['type']
        self.epochs = model_config['epochs']
        self.early_stop_epochs = model_config['early_stop_epochs']
        self.batch_size = model_config['batch_size']
        self.latent_dimension = model_config['latent_dimension']
        self.output_folder = config['output_path']
        self.features_folder_name = config['features_folder_name']
        self.language = config['language']
        self.x_train = x_train
        self.y_train = y_train

        # Create folder structure where to save the model
        self.full_path_output_folder = os.path.join(self.output_folder, self.name, self.features_folder_name)
        os.makedirs(self.full_path_output_folder, exist_ok=True)
        self.logs_folder_path = os.path.join(self.full_path_output_folder, 'logs/')
        os.makedirs(self.logs_folder_path, exist_ok=True)

    @abstractmethod
    def load_prediction_configuration(self, config):
        """
        It loads the configuration parameters for the prediction.
        :param config: a dictionary with the prediction configuration
        :return: instance will have the configuration parameters
        """
        self.output_folder = config['output_path']
        self.model_path = config['model_path']
        self.model_folder_name = config['model_folder_name']
        self.type = config['model_type']
        self.features_folder_name = config['features_folder_name']
        self.language = config['language']
        self.use_last_layer = config['use_last_layer']
        self.window_shift = config['window_shift']
        self.files_limit = config['files_limit']

    @abstractmethod
    def train(self):
        raise NotImplementedError('The model needs to overwrite the train method. The method should configure the '
                                  'learning process, callbacks and fit the model.')

    @abstractmethod
    def predict(self, x_test, x_test_ind, duration):
        """
        It predicts the output features for the test set (x_test) and output the predictions in text files using
        (x_test_ind).
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

        # Calculate predictions
        predictions = predictor.predict(x_test)

        # Create folder for predictions
        full_predictions_folder_path = os.path.join(self.output_folder,self.model_folder_name,
                                                    self.features_folder_name, self.language, (duration + 's'))
        os.makedirs(full_predictions_folder_path, exist_ok=True)

        # Create predictions text files
        total_files = create_prediction_files(predictions, x_test_ind, full_predictions_folder_path, self.window_shift,
                                              limit=self.files_limit)

        print('Predictions of {0} with duration {1}s: {2} files'.format(self.language, duration, total_files))
