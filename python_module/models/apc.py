"""
@date 26.02.2020
Autoregressive Predictive Coding model
[An unsupervised autoregressive model for speech representation learning]

This corresponds to a translation from the Pytorch implementation
(https://github.com/iamyuanchung/Autoregressive-Predictive-Coding) to Keras implementation
"""
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


