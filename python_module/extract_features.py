"""
    @author María Andrea Cruz Blandón
    @date 12.05.2020

    It extracts acoustic features from waveform files and saves the features in h5py files.
"""
import argparse
import glob
import os
import sys

import h5py
import librosa
import numpy as np
import numpy.matlib as mb

from read_configuration import read_configuration_json


def get_files(data_path):
    """
    It returns a list of tuples (file_name, file_size) of all .wav files in the specified folder. File names are ordered
    by file size from greater to smaller.
    :param data_path: folder path where the files are located
    :return: a list of tuples (file_name, file_size) ordered by file size
    """

    files_names = glob.iglob(os.path.join(data_path, '*.wav'))
    file_size_pairs = [(fname, os.stat(fname).st_size) for fname in files_names]
    ordered_files = sorted(file_size_pairs, key=lambda pair: pair[1], reverse=True)

    if len(ordered_files) == 0:
        raise Exception('There were no audio (.wav) files in: %s' % data_path)

    return ordered_files


def generate_datasets(file_list, val_split=0.2, max_size_outsiders=25000000, mode='diff'):
    """
    It creates two list of files, one for training set and the other for validation set. The validation set could
    include only outsider speakers or a mix of relative and outsider speakers.
    :param file_list: a list of tuples (file_name, file_size) ordered by file size.
    :param val_split: maximum percentage of data for validation set. default 0.2 (20%)
    :param max_size_outsiders: maximum size in bytes of outsider speakers files, default 25 Mb 10 min of audio,
    :param mode: a string stating the type of dataset to generate, diff is for a validation set with only outsiders and
                 mix is for a validation set with outsiders and relatives.
    :return: two lists, training and validation sets
    """
    total_size = sum([size for (file, size) in file_list])
    max_validation_size = int(total_size * val_split)

    training_set = []
    validation_set = []

    total_size_validation_set = 0
    if mode == 'diff':
        for (file_name, file_size) in file_list:
            if file_size > max_size_outsiders:
                training_set.append(file_name)
            else:
                if total_size_validation_set > max_validation_size:
                    training_set.append(file_name)
                else:
                    validation_set.append(file_name)
                    total_size_validation_set += file_size
    elif mode == 'mix':
        # Include at least one file with relatives.
        validation_set.append(file_list[0][0])
        total_size_validation_set += file_list[0][1]
        file_list = file_list[1:]
        file_list.reverse()
        for (file_name, file_size) in file_list:
            if total_size_validation_set > max_validation_size:
                training_set.append(file_name)
            else:
                validation_set.append(file_name)
                total_size_validation_set += file_size
    else:
        raise Exception('mode "%s" is not supported. Available modes: diff and mix.' % mode)
    return training_set, validation_set


def extract_mfcc(file_paths, window_length, window_shift, coefficients=13, deltas=True, deltas_deltas=True, cmvn=True):
    """
    It extracts MFCC coefficients of a set of files.
    :param file_paths: list of .wav file paths
    :param window_length: time in milliseconds of the window length
    :param window_shift: time in milliseconds of the window shift
    :param coefficients: number of mfcc coefficients to be extracted
    :param deltas: a boolean stating weather delta coefficients are required
    :param deltas_deltas: a boolean stating weather delta delta coefficients are required
    :param cmvn: a boolean stating weather Cepstrum Mean and Normalisation (CMVN) is required. (file-by-file)
    :return: a list of MFCC coefficients (numpy matrices with shape Frame X Coefficients) for each file.
    """
    features = []
    target_sampling_freq = 16000

    window_length_sample = int(target_sampling_freq * window_length)
    window_shift_sample = int(target_sampling_freq * window_shift)

    for idx, file in enumerate(file_paths):
        signal, sampling_freq = librosa.load(file, sr=target_sampling_freq)
        mfcc = librosa.feature.mfcc(signal, target_sampling_freq, n_mfcc=coefficients, n_fft=window_length_sample,
                                    hop_length=window_shift_sample)
        if deltas:
            mfcc_tmp = mfcc
            mfcc_deltas = librosa.feature.delta(mfcc_tmp)
            mfcc =np.concatenate([mfcc, mfcc_deltas])
            if deltas_deltas:
                mfcc_deltas_deltas = librosa.feature.delta(mfcc_tmp, order=2)
                mfcc = np.concatenate([mfcc, mfcc_deltas_deltas])

        mfcc = np.transpose(mfcc)
        # Replace zeros
        min_mfcc = np.min(np.abs(mfcc[np.nonzero(mfcc)]))
        mfcc = np.where(mfcc == 0, min_mfcc, mfcc)

        # Normalisation
        if cmvn:
            # mean = np.expand_dims(np.mean(mfcc, axis=0), 0)
            # std = np.expand_dims(np.std(mfcc, axis=0), 0)
            mean = mb.repmat(np.mean(mfcc, axis=0), mfcc.shape[0], 1)
            std = mb.repmat(np.std(mfcc, axis=0), mfcc.shape[0], 1)
            mfcc = np.divide((mfcc - mean), std)

            # mfcc = (mfcc - mean) / std

        features.append(mfcc)
        print('{}/{} file processed'.format(idx, len(file_paths)))
        print(mfcc.shape)
    return features


def create_h5py_file(output_path, file_paths, features, sample_length, duration=None, set_type='train'):
    """
    It creates the h5py file containing the acoustic features and indices to matching audio file.
    :param output_path: path where to save the h5py file
    :param file_paths: a list of string with the path of each audio
    :param features: a list of the acoustic features for each audio
    :param sample_length: numbers of frames per sample
    :param duration: duration of test files when set_type='test'
    :param set_type: string stating the type of the set (train, val, test)
    :return: a h5py file with features and indices of files
    """
    file_id = 0
    file_mapping = {}
    data = np.concatenate(features)
    frame_indices = []

    if set_type == 'test':
        assert duration is not None

    for file, feature in zip(file_paths, features):
        file_mapping[file_id] = file
        frames = feature.shape[0]
        idx = np.zeros((frames, 2))
        if set_type == 'test':
            idx[:, 0] = int(os.path.splitext(os.path.basename(file))[0])
        else:
            idx[:, 0] = file_id
        idx[:, 1] = np.arange(0, frames, 1)
        frame_indices.append(idx)

    indices = np.concatenate(frame_indices)

    total_frames = data.shape[0]
    n_feats = data.shape[1]
    extra_frames = sample_length - (total_frames % sample_length)

    if extra_frames:
        data = np.concatenate((data, np.zeros((extra_frames, n_feats))))
        idx = np.zeros((extra_frames, 2))
        idx[:, 0] = -1
        indices = np.concatenate((indices, idx))

    total_samples = int(data.shape[0]/sample_length)
    data = data.reshape((total_samples, sample_length, -1))
    indices = indices.reshape((total_samples, sample_length, -1))

    if set_type == 'test':
        with h5py.File(os.path.join(output_path, 'test_' + duration + 's' + '.h5'), 'w') as train_out_file:
            train_out_file.create_dataset('X_test_in', data=data)
            train_out_file.create_dataset('X_test_ind', data=indices)
    else:
        with h5py.File(os.path.join(output_path, set_type + '.h5'), 'w') as train_out_file:
            train_out_file.create_dataset('data', data=data)
            train_out_file.create_dataset('file_list', data=str(file_mapping))
            train_out_file.create_dataset('indices', data=indices)


def extract_features(config_path):
    """
        Extract the acoustic features for the specified files (languages, durations, train, and test)
        :param config_path: path to the JSON configuration file
        :return: input features in h5py files
        """
    # read configuration file
    config = read_configuration_json(config_path, False, False, True)['feature_extraction']
    method = config['method']
    for language in config['languages']:
        output_path = os.path.join(config['features_path'], language)
        os.makedirs(output_path, exist_ok=True)

        if config['train']:
            data_path = os.path.join(config['data_path'], language, 'train')
            train_paths, val_paths = generate_datasets(get_files(data_path), mode=config['mode'])
            train_feats = extract_mfcc(train_paths, method['window_length'], method['window_shift'],
                                       method['coefficients'], method['deltas'], method['deltas_deltas'],
                                       method['cmvn'])
            create_h5py_file(output_path, train_paths, train_feats, method['sample_length'], set_type='train')
            val_feats = extract_mfcc(val_paths, method['window_length'], method['window_shift'], method['coefficients'],
                                     method['deltas'], method['deltas_deltas'], method['cmvn'])
            create_h5py_file(output_path, val_paths, val_feats, method['sample_length'], set_type='val')
            print('Acoustic features extracted for %s training set' % language)
        if config['test']:
            data_path = os.path.join(config['data_path'], language, 'test')
            for duration in config['durations']:
                data_path = os.path.join(data_path, duration + 's')
                files = get_files(data_path)
                files = [filename for (filename, size) in
                         sorted(files, key=lambda pair: int(os.path.splitext(os.path.basename(pair[0]))[0]))]
                test_feats = extract_mfcc(files, method['window_length'], method['window_shift'],
                                          method['coefficients'], method['deltas'], method['deltas_deltas'],
                                          method['cmvn'])
                create_h5py_file(output_path, files, test_feats, method['sample_length'], duration, set_type='test')

            print('Acoustic features extracted for %s testing set, durations: ' % language +
                  ', '.join(config['durations']))


if __name__ == '__main__':
    # Call from command line
    parser = argparse.ArgumentParser(description='Script for extracting the acoustic features, the features are saved '
                                                 'in h5py files for later use.\nUsage: python extract_features.py '
                                                 '--config path_JSON_configuration_file')
    parser.add_argument('--config', type=str, default='./config.json')
    args = parser.parse_args()

    # Train model
    extract_features(args.config)
    sys.exit(0)