"""
@date 25.02.2020
Abstract class for neural networks models using Keras.
New models should inherit from this class so we can run training and prediction the same way for
all the models and independently.
"""

from abc import ABC, abstractmethod


class ModelBase(ABC):
    def __init__(self, config, x_train, y_train):
        # properties from configuration file
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
        super().__init__()

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self):
        raise NotImplementedError
