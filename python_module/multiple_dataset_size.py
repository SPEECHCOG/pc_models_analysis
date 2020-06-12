"""
    @date 27.05.2020
    It executes training of several models changing the dataset size
"""
import argparse
import glob
import json
import os
import sys

from train import train


def change_configuration_file(percentage, json_path='config.json'):
    """
    It reads json configuration file (config.json) and changes dataset_percentage parameter
    :param percentage int indicating the percentage of the dataset size to be used
    :param json_path string path of the json configuration file
    :return: it changes the file and returns the file name
    """
    with open(json_path) as f:
        config = json.load(f)

        config['training']['dataset_percentage'] = percentage

    # save json
    folder_path, file_name = os.path.split(json_path)
    output_file_path = os.path.join(folder_path, str(percentage) + '_' + file_name)
    with open(output_file_path, 'w') as f:
        json.dump(config, f, indent=2)
    return output_file_path


if __name__ == '__main__':
    # Call from command line
    parser = argparse.ArgumentParser(description='Script for running multiple times train.py script.'
                                                 '\nUsage: python multiple_dataset_size.py --percentage <percentage> '
                                                 '--config config.json')
    parser.add_argument('--percentage', type=int, default=100)
    parser.add_argument('--config', type=str, default='./config.json')
    args = parser.parse_args()

    new_config_file = change_configuration_file(args.percentage, json_path=args.config)
    # Train model
    train(new_config_file)
    os.remove(new_config_file)
    sys.exit(0)
