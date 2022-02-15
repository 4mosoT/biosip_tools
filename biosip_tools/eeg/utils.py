from .timeseries import EEGSeries
import numpy as np


def window_data_loader(eegs: EEGSeries, batch_size: int = 32, window_size: float = 1, infinity: bool = False, labels: np.ndarray = None, epochs=1, shuffle: bool = False, return_subjects=False, stride: float = None) -> tuple:
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
    :param stride: stride in seconds, defaults to None
    :type stride: float, optional
    :yield: batch and labels if labels are provided
    :rtype: tuple
    """

    window_sample = int(window_size * eegs.sample_rate)
    data = np.lib.stride_tricks.sliding_window_view(
        eegs.data, window_sample, axis=-1)
    if stride is not None:
        data = data[:, :, :: int(eegs.sample_rate * stride), :]
    else:
        data = data[:, :, ::window_sample, :]
    data = data.reshape(-1, data.shape[1], data.shape[-1])

    if return_subjects:
        batch_size = data.shape[0] // eegs.data.shape[0]

    if labels is not None:
        new_labels = np.repeat(labels, data.shape[0] // eegs.data.shape[0])

    for epoch in range(epochs):
        done_epoch = None
        batch_number = 0

        if shuffle:
            p = np.random.permutation(len(data))
            data = data[p]
            if labels is not None:
                new_labels = new_labels[p]

        while done_epoch is None:
            data_batch = data[batch_number *
                                batch_size:(batch_number + 1) * batch_size]
            if labels is not None:
                label_batch = new_labels[batch_number *
                                            batch_size:(batch_number + 1) * batch_size]
            batch_number += 1
            if batch_number  * batch_size >= data.shape[0]:
                done_epoch = epoch
            yield data_batch, label_batch if labels is not None else None, done_epoch
