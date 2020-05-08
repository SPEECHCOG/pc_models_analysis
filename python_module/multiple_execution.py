"""
@date 31.03.2020

It changes the json configuration file according to the language code given, and run the training accordingly:
1: English
2: French
3: Mandarin
4: Lang1 (German)
5: Lang2 (Wolof)
"""
import argparse
import json
import os
import sys

from train import train


def change_configuration_file(lang_code, json_path='config.json'):
    """
    It reads json configuration file (config.json) and changes the language according to the language code provided
    :param lang_code int stating the language to be used.
    :param json_path string path of the json configuration file
    :return: it changes the file and run train or predict with the new file
    """
    with open(json_path) as f:
        configuration = json.load(f)
        previous_lang = configuration['training']['language']
        if lang_code == 1:
            new_lang = 'english'
        elif lang_code == 2:
            new_lang = 'french'
        elif lang_code == 3:
            new_lang = 'mandarin'
        elif lang_code == 4:
            new_lang = 'LANG1'
        elif lang_code == 5:
            new_lang = 'LANG2'
        else:
            Exception('Only options: 1-5 (english, french, mandarin, LANG1, LANG2)')

        # Update info
        configuration['training']['train_in'][0] = configuration['training']['train_in'][0].replace(
            previous_lang, new_lang)
        configuration['training']['train_out'][0] = configuration['training']['train_out'][0].replace(
            previous_lang, new_lang)
        configuration['training']['language'] = new_lang

    # save json
    folder_path, file_name = os.path.split(json_path)
    output_file_path = os.path.join(folder_path, new_lang + '_' + file_name)
    with open(output_file_path, 'w') as f:
        json.dump(configuration, f, indent=2)
    return output_file_path

if __name__ == '__main__':
    # Call from command line
    parser = argparse.ArgumentParser(description='Script for running multiple times train.py script.'
                                                 '\nUsage: python multiple_execution.py '
                                                 '<language_code>.\nLanguage code:\n\t1: English\n\t2:French\n\t3:'
                                                 'Mandarin\n\t4:Lang1(German)\n\t5:Lang2(Wolof)')
    parser.add_argument('--lang_code', type=int, default=3)
    parser.add_argument('--config', type=str, default='./config.json')
    args = parser.parse_args()

    new_config_file = change_configuration_file(args.lang_code, json_path=args.config)
    # Train model
    train(new_config_file)
    os.remove(new_config_file)
    sys.exit(0)