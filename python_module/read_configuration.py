"""
@date 24.02.2020

It reads a configuration in the JSON format for both training and prediction. It validates the fields of the
configurations. Training input/output features and test set are load from .mat files (MatLab)
"""
import json
import os
import scipy.io

# Fields of configuration (name, type, default value, required?).
TRAINING_CONFIG = [
    ('train_in', list, None, True),
    ('train_out', list, None, True),
    ('output_path', str, None, True),
    ('features_folder_name', str, None, True),
    ('language', str, 'mandarin', False),
    ('model', dict, None, True)
]

TRAINING_MODEL_CONFIG = [
    ('name', str, None, True),
    ('type', str, None, True),
    ('epochs', int, 100, False),
    ('early_stop_epochs', int, 50, False),
    ('batch_size', int, 32, False),
    ('latent_dimension', int, None, True)
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
    ('files_limit', int, 10, False)
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
    for field in ['train_in', 'train_out']:
        if len(config['training'][field]) != 2:
            raise Exception('%s should have 2 strings in the format: [path, name_of_variable]' % field[0])
        if type(config['training'][field][0]) != str or type(config['training'][field][1]) != str:
            raise Exception('%s should have only string items' % field)

    # Validate input files
    for path in [config['training']['train_in'][0], config['training']['train_out'][0]]:
        if not os.path.exists(path):
            raise Exception('The file does not exist: %s' % path)

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
    test_paths = [os.path.join(config['prediction']['test_set'][0], ('test_' + d + 's.mat'))
                  for d in config['prediction']['durations']]
    for path in [config['prediction']['model_path']] + test_paths:
        if not os.path.exists(path):
            raise Exception('The file does not exist: %s' % path)

    return config


def load_training_features(train_in, train_out):
    """
    It reads .mat files and output the tensors
    :param train_in: configuration of training input features
    :param train_out: configuration of training output features
    :return: two numpy arrays: training input features and training output features
    """

    x_train = scipy.io.loadmat(train_in[0])[train_in[1]]
    y_train = scipy.io.loadmat(train_out[0])[train_out[1]]

    return x_train, y_train


def load_test_set(test_set, duration):
    """
    It reads a .mat file with the test set features plus the indices to match each frame to the source
    file.
    :param test_set: configuration of the test set
    :param duration: string specifying the test set duration (possible values 1, 10, 120)
    :return: two numpy arrays: test input features and indices of frames
    """
    matlab_tensors = scipy.io.loadmat(os.path.join(test_set[0], ('test_' + duration + 's.mat')))
    x_test = matlab_tensors[test_set[1]]
    x_test_ind = matlab_tensors[test_set[2]]

    return x_test, x_test_ind


def read_configuration_json(json_path, is_training, is_prediction):
    """
    It reads a JSON file that contains the configuration for both training and prediction.
    The validation of training and prediction configurations will be determined by the booleans is_training and
    is_predictions. It returns the validated configuration.
    An exception is raised if the path does not exists.
    :param json_path: string specifying the path where the JSON configuration file is located
    :param is_training: boolean stating if the configuration will be used for training a model
    :param is_prediction: boolean stating if the configuration will be used for prediction
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

        return configuration
