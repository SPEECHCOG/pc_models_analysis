"""
    @author Maria Andrea Cruz Blandon
    @date 19.02.2020

    Create text files for submission from features predicted with a model.
"""
import os
import numpy as np


def calculate_timestamps(n_frames, window_shift):
    """
    It calculates timestamps for a number of total frames (n_frames) using the window shift (window_shift)
    :param n_frames: Total number of frames
    :param window_shift: Time in seconds for the window shift
    :return: a numpy array with the timestamps for all the frames
    """
    timestamps = np.arange(window_shift/2, n_frames*window_shift, window_shift)

    return timestamps


def create_output_file(features, file_path, window_shift):
    """
    It creates one output file given the matrix of features
    :param features: numpy array with the predicted features
    :param file_path: path of the output file
    :param window_shift: window shift, in seconds, used to calculate frames.
    :return: a text file with the predictions is created
    """
    n_frames = features.shape[0]
    timestamps = calculate_timestamps(n_frames, window_shift)

    # save txt file
    np.savetxt(file_path, np.c_[timestamps, features], delimiter=' ', fmt='%.5f')


def create_prediction_files(prediction, indices, folder_path, window_shift, limit=None):
    """
    It maps the samples with the respective source files (Samples may contain frames from different audio files).
    This mapping needs to be done before exporting the predicted features to text files.
    :param prediction: a numpy array with all the features predicted for each frame per samples.
                       The dimension is samples x time-steps x features.
    :param indices: a numpy array with the indices (number of frame in the source audio) for each sample. The dimension
                    is samples x time-steps x 2 (where the first number is the source audio identifier, and the second
                    one is the number of the frame in the audio)
    :param folder_path: path to the folder where the files should be saved.
    :param window_shift:
    :param limit: the number of maximum files to create
    :return: The total number of files created
    """
    # Verify the arrays have the same two first dimensions
    assert prediction.shape[0] == indices.shape[0]
    assert prediction.shape[1] == indices.shape[1]

    samples = prediction.shape[0]
    time_steps = prediction.shape[1]

    # Flatten the arrays
    # Another way to do this: arr.reshape(-1, arr.shape[-1])
    prediction = prediction.reshape(samples*time_steps, prediction.shape[-1])
    indices = indices.reshape(samples*time_steps, indices.shape[-1])

    # A list of tuples indicating the indices of each source file. (init, end) init and end indices in the flatted array
    file_indices = []
    file_names = []

    prev = indices[0, 0]  # first file id
    init = 0

    if not isinstance(limit, int):
        limit = -1  # we need to create all the files

    total_frames = indices.shape[0]
    for i in range(total_frames):
        # check limit
        if len(file_indices) == limit:
            break  # we reach the limit

        # Update file indices if the index changed or if we have reached the last element in the array
        if indices[i, 0] != prev:
            file_indices.append((init, i))
            file_names.append(prev)
            prev = indices[i, 0]
            init = i
        if i == total_frames - 1:
            file_names.append(prev)
            file_indices.append((init, total_frames))

    # writes files
    for i in range(len(file_indices)):
        idx_init = file_indices[i][0]
        idx_end = file_indices[i][1]
        if file_names[i] != -1:
            # frames marked with -1 as the file id, are reset/padding frames. Therefore won't be included into the
            # output
            create_output_file(prediction[idx_init:idx_end, :],
                               os.path.join(folder_path, '%d.txt' % file_names[i]), window_shift)

    return len(file_indices)