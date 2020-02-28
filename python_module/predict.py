"""
@date 25.02.2020

It reads a configuration file and calculate the predictions for a test set using the specified model.
"""
import argparse
import sys

from models.apc import APCModel
from models.autoencoder import Autoencoder
from read_configuration import read_configuration_json, load_test_set


def predict(config_path):
    """
    Predict the output features of a test set using the configuration parameters provided in the configuration file
    :param config_path: path to the JSON configuration file
    :return: predictions in the text format
    """
    # read configuration file
    config = read_configuration_json(config_path, False, True)['prediction']

    # Use correct model
    model_type = config['model_type']

    if model_type == 'autoencoder':
        model = Autoencoder()
    elif model_type == 'apc':
        model = APCModel()
    else:
        raise Exception('The model type "%s" is not supported' % model_type)

    model.load_prediction_configuration(config)

    for duration in config['durations']:
        x_test, x_test_ind = load_test_set(config['test_set'], duration)
        model.predict(x_test, x_test_ind, duration)

    print('Predictions for ' + config['language'] + ' with durations (' + ', '.join(config['durations']) + ') are '
          'finished')


if __name__ == '__main__':
    # Call from command line
    parser = argparse.ArgumentParser(description='Script for calculate the predictions of a model from a test set. It '
                                                 'uses a configuration file in JSON format.\nUsage: python predict.py '
                                                 '--config path_JSON_configuration_file')
    parser.add_argument('--config', type=str, default='./config.json')
    args = parser.parse_args()

    # Train model
    predict(args.config)
    sys.exit(0)
