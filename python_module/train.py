"""
@date 25.02.2020

It trains a model according to the configuration given
"""
import argparse
import sys

from models.apc import APCModel
from models.cpc import CPCModel
from read_configuration import read_configuration_json, load_training_features, load_python_training_features


def train(config_path):
    """
    Train a neural network model using the configuration parameters provided
    :param config_path: the path to the JSON configuration file.
    :return: The trained model is saved in a h5 file
    """

    # read configuration file
    config = read_configuration_json(config_path, True, False)['training']

    # Obtain input/output features to train the model
    if config['features_type'] == 'matlab':
        x_train, y_train = load_training_features(config['train_in'], config['train_out'])
        x_val, y_val = None, None
    else:
        input_feats = config['input_features']
        shift = (config['model']['type'] == 'apc' or config['model']['type'] == 'convpc')
        x_train, x_val, y_train, y_val = load_python_training_features(input_feats['train'],
                                                                       input_feats['validation'],
                                                                       steps=input_feats['shift'],
                                                                       shift=shift)

    # Use correct model
    model_type = config['model']['type']

    if model_type == 'apc':
        model = APCModel()
    elif model_type == 'cpc':
        model = CPCModel()
    else:
        raise Exception('The model type "%s" is not supported' % model_type)

    model.load_training_configuration(config, x_train, y_train, x_val, y_val)
    model.train()

    print('Training of model "%s" finished' % model_type)


if __name__ == '__main__':
    # Call from command line
    parser = argparse.ArgumentParser(description='Script for training a model from the ones available in the python '
                                                 'module using a JSON configuration file. \nUsage: python train.py '
                                                 '--config path_JSON_configuration_file')
    parser.add_argument('--config', type=str, default='./config.json')
    args = parser.parse_args()

    # Train model
    train(args.config)
    sys.exit(0)
