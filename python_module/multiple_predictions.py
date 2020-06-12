"""
    @date 18.05.2020
    It executes the prediction for each epoch in a run (10 models per run, 1 per each 10 epochs)
"""
import argparse
import glob
import json
import os
import sys

from predict import predict


def change_configuration_file(epoch_id, json_path='config.json'):
    """
    It reads json configuration file (config.json) and changes the model path, and features folder name according to
    the number of epoch.
    :param epoch_id int indicating the epoch number
    :param json_path string path of the json configuration file
    :return: it changes the file and return the file name
    """
    with open(json_path) as f:
        config = json.load(f)
        lang = config['prediction']['language']
        folder_models = os.path.split(config['prediction']['model_path'])[0]
        model_path = glob.glob(os.path.join(folder_models, lang + '*-' + str(epoch_id) + '*.h5'))[0]

        config['prediction']['model_path'] = model_path
        config['prediction']['features_folder_name'] = str(epoch_id)

        val_split = 'mix' if 'mix' in folder_models else 'diff'

    # save json
    folder_path, file_name = os.path.split(json_path)
    output_file_path = os.path.join(folder_path, lang + '-' + val_split + '-' + str(epoch_id) + '_' + file_name)
    with open(output_file_path, 'w') as f:
        json.dump(config, f, indent=2)
    return output_file_path


if __name__ == '__main__':
    # Call from command line
    parser = argparse.ArgumentParser(description='Script for running multiple times train.py script.'
                                                 '\nUsage: python multiple_predictions.py --epoch_id <id> --config '
                                                 'config.json')
    parser.add_argument('--epoch_id', type=int, default=20)
    parser.add_argument('--config', type=str, default='./config.json')
    args = parser.parse_args()

    new_config_file = change_configuration_file(args.epoch_id, json_path=args.config)
    # Train model
    predict(new_config_file)
    os.remove(new_config_file)
    sys.exit(0)
