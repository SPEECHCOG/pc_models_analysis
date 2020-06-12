"""
@date 24.02.2020

It reads a configuration in the JSON format for both training and prediction. It validates the fields of the
configurations. Training input/output features and test set are load from .mat files (MatLab)
"""
import json
import math
import os

import h5py
import scipy.io
import numpy as np

# Fields of configuration (name, type, default value, required?).
TRAINING_CONFIG = [
    ('train_in', list, None, True),
    ('train_out', list, None, True),
    ('output_path', str, None, True),
    ('features_folder_name', str, None, True),
    ('language', str, 'mandarin', False),
    ('model', dict, None, True),
    ('input_features', dict, None, False),
    ('features_type', str, 'matlab', False),
    ('statistical_analysis', dict, None, False),
    ('dataset_percentage', int, 100, False)
]

TRAINING_MODEL_CONFIG = [
    ('name', str, None, True),
    ('type', str, None, True),
    ('epochs', int, 100, False),
    ('early_stop_epochs', int, 50, False),
    ('batch_size', int, 32, False),
    ('latent_dimension', int, None, True),
    ('apc', dict, None, False),
    ('cpc', dict, None, False)
]

INPUT_FEATURES_CONFIG =[
    ('train', str, None, True),
    ('validation', str, None, True),
    ('shift', int, 5, False)
]

STATISTICAL_ANALYSIS_CONFIG = [
    ('system', str, None, True),
    ('model_id', int, None, True),
    ('period', int, 10, False)
]

APC_CONFIG = [
    ('prenet', bool, False, False),
    ('prenet_layers', int, 0, False),
    ('prenet_dropout', float, 0, False),
    ('prenet_units', int, 0, False),
    ('rnn_layers', int, 3, False),
    ('rnn_dropout', float, 0, False),
    ('rnn_units', int, 512, False),
    ('residual', bool, True, False),
    ('learning_rate', float, 0.001, False)
]

CPC_CONFIG = [
    ("encoder_layers", int, 5, False),
    ("encoder_units", int, 512, False),
    ("encoder_dropout", float, 0.2, False),
    ("gru_units", int, 256, False),
    ("dropout", float, 0.2, False),
    ("neg", int, 10, False),
    ("steps", int, 12, False),
    ("learning_rate",float, 0.001, False)
]

PREDICTION_CONFIG = [
    ('output_path', str, None, True),
    ('model_path', str, None, True),
    ('model_folder_name', str, None, True),
    ('model_type', str, None, True),
    ('features_folder_name', str, None, True),
    ('test_set', list, None, True),
    ('language', str, 'mandarin', False),
    ('durations', list, ['10'], False),
    ('use_last_layer', bool, False, False),
    ('window_shift', float, 0.01, False),
    ('files_limit', int, -1, False),
    ('use_pca', bool, True, False),
    ('save_matlab', bool, False, False),
    ('features_type', str, 'matlab', False)
]

FEATURE_EXTRACTION_CONFIG = [
    ('data_path', str, None, True),
    ('features_path', str, None, True),
    ('languages', list, ['mandarin'], False),
    ('durations', list, ['10'], False),
    ('method', dict, None, True),
    ('mode', str, 'diff', False),
    ('train', bool, True, False),
    ('test', bool, True, False)
]

METHOD_FEATURE_EXTRACTION = [
    ('name', str, 'mfcc', False),
    ('window_length', float, 0.025, False),
    ('window_shift', float, 0.01, False),
    ('deltas', bool, True, False),
    ('deltas_deltas', bool, True, False),
    ('cmvn', bool, True, False),
    ('coefficients', int, 13, False),
    ('sample_length', int, 200, False)
]


LANGUAGES = ['english', 'mandarin', 'french', 'LANG1', 'LANG2']
DURATIONS = ['1', '10', '120']


def validate_fields(config, config_definition):
    """
    It validates a configuration against the required information (config_definition). It a required field is not
    provided, or the type of the field is not as the required an exception is raised. If non-required fields are not
    given, they are set to the default value
    :param config: a dictionary with the configuration
    :param config_definition: a list of fields definitions
    :return: It modifies the config dictionary
    """
    current_fields = list(config.keys())

    for field in config_definition:
        if field[0] not in current_fields:
            if field[3]:
                # It is a required field
                raise Exception('The training configuration is missing required field "%s"' % field[0])
            else:
                # Create field with default value
                config[field[0]] = field[2]
        else:
            # The field was provided.
            if type(config[field[0]]) != field[1]:
                raise Exception('The field "%s" should be of type %s' % (field[0], str(field[1])))


def validate_training(config):
    """
    It validates the training configuration data is in the right format, that paths exist, and assigns default
    values if non-required fields are not given.
    :param config: a dictionary with the configuration
    :return: The configuration with custom and default values where needed if the validation is satisfactory,
             otherwise an exception will be raised.
    """
    # Validate training fields
    validate_fields(config['training'], TRAINING_CONFIG)
    # Validate model fields
    validate_fields(config['training']['model'], TRAINING_MODEL_CONFIG)

    # Validate language
    if config['training']['language'] not in LANGUAGES:
        raise Exception('Only ' + ', '.join(LANGUAGES) + ' are supported.')

    # Validate training files fields
    if config['training']['features_type'] == 'matlab':
        for field in ['train_in', 'train_out']:
            if len(config['training'][field]) != 2:
                raise Exception('%s should have 2 strings in the format: [path, name_of_variable]' % field[0])
            if type(config['training'][field][0]) != str or type(config['training'][field][1]) != str:
                raise Exception('%s should have only string items' % field)

        # Validate input files
        for path in [config['training']['train_in'][0], config['training']['train_out'][0]]:
            if not os.path.exists(path):
                raise Exception('The file does not exist: %s' % path)
    else:
        validate_fields(config['training']['input_features'], INPUT_FEATURES_CONFIG)
        for path in [config['training']['input_features']['train'], config['training']['input_features']['validation']]:
            if not os.path.exists(path):
                raise Exception('The file does not exist: %s' % path)

    if config['training']['statistical_analysis'] is not None:
        validate_fields(config['training']['statistical_analysis'], STATISTICAL_ANALYSIS_CONFIG)

    # Validate models configuration
    # apc
    if config['training']['model']['type'] == 'apc':
        if not config['training']['model']['apc']:
            # The model is apc but there were not parameters, we need to create default parameters
            config['training']['model']['apc'] = {}
        # validate parameters for apc model
        validate_fields(config['training']['model']['apc'], APC_CONFIG)
    elif config['training']['model']['type'] == 'cpc':
        if not config['training']['model']['cpc']:
            config['training']['model']['cpc'] = {}
        validate_fields(config['training']['model']['cpc'], CPC_CONFIG)

    return config


def validate_prediction(config):
    """
    It validates the prediction configuration data is in the right format, that paths exist, and assigns defaults
    values if non-required fields are not provided.
    :param config: a dictionary with the configuration
    :return: The configuration with defaults values if needed. An exception will be raised if the validation is not
             satisfactory
    """

    # Validate prediction fields
    validate_fields(config['prediction'], PREDICTION_CONFIG)

    # Validate test_set field
    if len(config['prediction']['test_set']) != 3:
        raise Exception('test_set should have 3 strings in the format: [path, name_of_variable_features, '
                        'name_of_variable_indices]')

    if type(config['prediction']['test_set'][0]) != str or type(config['prediction']['test_set'][1]) != str or \
            type(config['prediction']['test_set'][2]) != str:
        raise Exception('test_set should have only string items')

    # Validate language
    if config['prediction']['language'] not in LANGUAGES:
        raise Exception('Only ' + ', '.join(LANGUAGES) + ' are supported.')

    # Validate durations
    for duration in config['prediction']['durations']:
        if duration not in DURATIONS:
            raise Exception('Only durations of: ' + ', '.join(DURATIONS) + ' are supported.')

    # Validate paths
    if config['prediction']['features_type'] == 'matlab':
        test_paths = [os.path.join(config['prediction']['test_set'][0], ('test_' + d + 's.mat'))
                      for d in config['prediction']['durations']]
    else:
        test_paths = [os.path.join(config['prediction']['test_set'][0], ('test_' + d + 's.h5'))
                      for d in config['prediction']['durations']]
    for path in [config['prediction']['model_path']] + test_paths:
        if not os.path.exists(path):
            raise Exception('The file does not exist: %s' % path)

    return config


def validate_feature_extraction(config):
    """
    It validates the feature extraction configuration data is in the right format, that paths exist, and assigns
    defaults values if non-required fields are not provided.
    :param config: a dictionary with the configuration
    :return: The configuration with defaults values if needed. An exception will be raised if the validation is not
             satisfactory
    """

    # Validate feature extraction fields
    validate_fields(config['feature_extraction'], FEATURE_EXTRACTION_CONFIG)

    # Validate language
    for language in config['feature_extraction']['languages']:
        if language not in LANGUAGES:
            raise Exception('Only ' + ', '.join(LANGUAGES) + ' are supported.')

    # Validate durations
    for duration in config['feature_extraction']['durations']:
        if duration not in DURATIONS:
            raise Exception('Only durations of: ' + ', '.join(DURATIONS) + ' are supported.')

    # Validate method
    validate_fields(config['feature_extraction']['method'], METHOD_FEATURE_EXTRACTION)

    return config


def load_python_training_features(train, val, steps=None, shift=False):
    """
    It reads h5py files containing the input features for training
    :param train: path of input features training set
    :param val: path of input features validation set
    :param shift: set True if target features should be shifted
    :return: train and validation variables
    """
    with h5py.File(train, 'r') as file:
        x_train = np.array(file['data'])
    with h5py.File(val, 'r') as file:
        x_val = np.array(file['data'])

    y_train, y_val = None, None
    if shift:
        assert steps is not None
        sample_length = x_train.shape[1]
        n_feats = x_train.shape[-1]

        x_train_flat = x_train.reshape((-1, n_feats))
        x_train = x_train_flat[:-steps, :]
        y_train = x_train_flat[steps:, :]

        x_val_flat = x_val.reshape((-1, n_feats))
        x_val = x_val_flat[:-steps, :]
        y_val = x_val_flat[steps:, :]

        total_samples_tr = int(x_train.shape[0]/sample_length)
        total_frames_tr = total_samples_tr  * sample_length
        x_train = x_train[:total_frames_tr, :]
        y_train = y_train[:total_frames_tr, :]

        total_samples_vl = int(x_val.shape[0]/sample_length)
        total_frames_vl = total_samples_vl * sample_length
        x_val = x_val[:total_frames_vl, :]
        y_val = y_val[:total_frames_vl, :]

        x_train = x_train.reshape((total_samples_tr, sample_length, -1))
        y_train = y_train.reshape((total_samples_tr, sample_length, -1))

        x_val = x_val.reshape((total_samples_vl, sample_length, -1))
        y_val = y_val.reshape((total_samples_vl, sample_length, -1))

    return x_train, x_val, y_train, y_val


def load_training_features(train_in, train_out):
    """
    It reads .mat files and output the tensors
    :param train_in: configuration of training input features
    :param train_out: configuration of training output features
    :return: two numpy arrays: training input features and training output features
    """
    # Training file could be saved in scipy or h5py format.
    try:
        with h5py.File(train_in[0], 'r') as train_in_file:
            x_train = np.array(train_in_file[train_in[1]]).transpose()
    except OSError:
        x_train = scipy.io.loadmat(train_in[0])[train_in[1]]

    try:
        with h5py.File(train_out[0], 'r') as train_out_file:
            y_train = np.array(train_out_file[train_out[1]]).transpose()
    except OSError:
        y_train = scipy.io.loadmat(train_out[0])[train_out[1]]

    return x_train, y_train


def load_test_set(test_set, duration, features_type='matlab'):
    """
    It reads a .mat file with the test set features plus the indices to match each frame to the source
    file.
    :param test_set: configuration of the test set
    :param duration: string specifying the test set duration (possible values 1, 10, 120)
    :return: two numpy arrays: test input features and indices of frames
    """
    if features_type == 'matlab':
        try:
            with h5py.File(os.path.join(test_set[0], ('test_' + duration + 's.mat')), 'r') as test_features:
                # We need to transpose because order in matlab is different than in python and therefore the dimensions
                # are transposed. For example: x (n,m,l) in matlab will be x (l,m,n) in python
                # https://stackoverflow.com/a/39264426
                x_test = np.array(test_features[test_set[1]]).transpose()
                x_test_ind = np.array(test_features[test_set[2]]).transpose()
        except OSError:
            x_test = scipy.io.loadmat(os.path.join(test_set[0], ('test_' + duration + 's.mat')))[test_set[1]]
            x_test_ind = scipy.io.loadmat(os.path.join(test_set[0], ('test_' + duration + 's.mat')))[test_set[2]]
    else:
        with h5py.File(os.path.join(test_set[0], ('test_' + duration + 's.h5')), 'r') as test_file:
            x_test = np.array(test_file[test_set[1]])
            x_test_ind = np.array(test_file[test_set[2]])

    return x_test, x_test_ind


def read_configuration_json(json_path, is_training, is_prediction, is_feature_extraction=False):
    """
    It reads a JSON file that contains the configuration for both training and prediction.
    The validation of training and prediction configurations will be determined by the booleans is_training and
    is_predictions. It returns the validated configuration.
    An exception is raised if the path does not exists.
    :param json_path: string specifying the path where the JSON configuration file is located
    :param is_training: boolean stating if the configuration will be used for training a model
    :param is_prediction: boolean stating if the configuration will be used for prediction
    :param is_feature_extraction: boolean stating if the configuration will be used for feature extraciton
    :return: a dictionary with the configuration parameters
    """

    with open(json_path) as f:
        configuration = json.load(f)

        if is_training:
            if 'training' not in configuration:
                raise Exception('The configuration should have "training" field for training configuration.')
            else:
                configuration = validate_training(configuration)
        if is_prediction:
            if 'prediction' not in configuration:
                raise Exception('The configuration should have "prediction" field for prediction configuration.')
            else:
                configuration = validate_prediction(configuration)
        if is_feature_extraction:
            if 'feature_extraction' not in configuration:
                raise Exception('The configuration should have "feature_extraction" field for feature extraction '
                                'configuration')
            else:
                configuration = validate_feature_extraction(configuration)

        return configuration
