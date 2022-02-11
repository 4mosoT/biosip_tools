from .timeseries import EEGSeries
import numpy as np


def window_data_loader(eegs: EEGSeries, batch_size: int = 32, window_size: float = 1, infinity: bool = False, labels: np.ndarray = None, epochs=1, shuffle: bool = False, return_subjects=False) -> tuple:
    """[summary]

    :param eegs: [description]
    :type eegs: EEGSeries
    :param batch_size: batch size, defaults to 32
    :type batch_size: int, optional
    :param window_size: eeg window in seconds, defaults to 1
    :type window_size: float, optional
    :param infinity: whether the loop should be infinity, defaults to False
    :type infinity: bool, optional
    :param labels: Labels per subject, defaults to None
    :type labels: np.ndarray, optional
    :param epochs: number of epochs, defaults to 1
    :type epochs: int, optional
    :param shuffle: whether to shuffle the data, defaults to False
    :type shuffle: bool, optional
    :param return_subjects: batch size correspond to number of windows per subject, defaults to False
    :type return_subjects: bool, optional
    :yield: batch and labels if labels are provided
    :rtype: tuple
    """

    # TODO: Implement overlapping (stride)

    window_sample = int(window_size * eegs.sample_rate)
    n_windows = int(np.floor(eegs.data.shape[2] / window_sample))
    trim_to_fit = eegs.data.shape[2] % window_sample
    #print("Removing {} samples from the start of the data".format(trim_to_fit))
    data = eegs.data[:, :, trim_to_fit:].reshape(
        *eegs.data.shape[:2], n_windows, -1)
    data = data.reshape(-1, data.shape[1], data.shape[-1])
    if return_subjects:
        batch_size = n_windows

    if labels is not None:
        new_labels = np.repeat(labels, data.shape[0] // eegs.data.shape[0])

    batch_number = 0
    for epoch in range(epochs):
        done_epoch = None

        if shuffle:
            p = np.random.permutation(len(data))
            data = data[p]
            if labels is not None:
                new_labels = new_labels[p]

        while done_epoch is None:
            if batch_number * batch_size >= data.shape[0]:
                batch_number = 0
                done_epoch = epoch
                data_batch = data[batch_number * batch_size:]
                if labels is not None:
                    label_batch = new_labels[batch_number * batch_size:]

            data_batch = data[batch_number *
                              batch_size:(batch_number + 1) * batch_size]
            if labels is not None:
                label_batch = new_labels[batch_number *
                                         batch_size:(batch_number + 1) * batch_size]
            batch_number += 1
            yield data_batch, label_batch if labels is not None else None, done_epoch
