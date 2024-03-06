from operator import sub
import numpy as np
import mne
from scipy import signal
import matplotlib.pyplot as plt
from .constants import EEG_BANDS
import ast


class EEGSeries():

    def __init__(self, data: np.ndarray = None, path: str = None, sample_rate: int = 500, subjects_info: str = None) -> None:
        """Class for EEG time series.

        :param data: EEG data.
        :type data: np.ndarray
        :param path: Path to .npy array. Expected shape is (n_subjects, n_channels, n_samples)
        :type path: str
        :param sample_rate: sample rate, defaults to 500
        :type sample_rate: int, optional
        :param subjects_info: path to subjects txt file
        :type subjects_info: str, defaults to None
        """
        assert (data is not None or path is not None)
        self.data = np.load(path) if path is not None else data
        assert (len(self.data.shape) == 3)
        self.sample_rate = sample_rate
        self.subjects_order = None

        if subjects_info is not None:
            with open(subjects_info, 'r') as f:
                self.subjects_order = [x[:3]+"_"+x[-5]
                                       for x in ast.literal_eval(f.read())]
        self.shape = self.data.shape

    def __getitem__(self, key):
        return self.data[key]

    def apply_cheby_filter(self, lowcut: float, highcut: float, order: int = 6, rs: float = 40, plot_response=False):
        """Apply a Chebyshev II filter to the EEG data.

        :param lowcut: Lower pass-band edge.
        :type lowcut: float
        :param highcut: Upper pass-band edge.
        :type highcut: float
        :param order: [description], defaults to 6
        :type order: int, optional
        :param rs: [description], defaults to 40
        :type rs: float, optional
        :param plot_response: [description], defaults to False
        :type plot_response: bool, optional
        :return: Filtered data
        :rtype: EEGSeries
        """

        nyq = 0.5 * self.sample_rate
        low = lowcut / nyq
        high = highcut / nyq
        sos = signal.cheby2(order, rs, [low, high], btype='band', output='sos')
        if plot_response:
            w, h = signal.sosfreqz(sos)
            plt.plot((self.sample_rate * 0.5 / np.pi)
                     * w, 20 * np.log10(abs(h)))
            plt.title('Frequency response (rs={})'.format(rs))
            plt.xlabel('Frequency')
            plt.ylabel('Amplitude [dB]')
            plt.margins(0, 0.1)
            plt.grid(which='both', axis='both')
            plt.axvline(lowcut, color='red')
            plt.axvline(highcut, color='red')
            plt.show()
        return EEGSeries(data=signal.sosfilt(sos, self.data))

    def fir_filter(self, l_freq: float, h_freq: float, verbose=False, **kwargs) -> np.ndarray:
        """Apply a FIR filter to the EEG data. Accepts arguments for mne.filter.filter_data.

        :param l_freq: Lower pass-band edge.
        :type l_freq: float
        :param h_freq: Upper pass-band edge.
        :type h_freq: float
        :return: Filtered data
        :rtype: EEGSeries
        """
        return EEGSeries(data=mne.filter.filter_data(self.data, self.sample_rate, l_freq, h_freq, verbose=verbose, **kwargs))

    def clip_and_normalize(self, n_deviations: int = 3):
        """Clip and normalize the EEG data.

        :param n_deviations: Number of standard deviations to clip at, defaults to 3
        :type n_deviations: int, optional
        :return: Normalized data
        :rtype: EEGSeries
        """
        shape = self.data.shape
        p = self.data.reshape(-1, shape[-1])

        for index, x in enumerate(p):
            m = np.mean(x)
            s = np.std(x)
            x = np.clip(x, m - n_deviations * s, m + n_deviations * s)
            p[index] = (x - x.min()) / (x.max() - x.min())

        return EEGSeries(data=p.reshape(shape))

    def fir_filter_bands(self, **kwargs) -> dict:
        """Return a dictionary of filtered EEG data. 

        :return: Dictionary of filtered data. {band_name: data}
        :rtype: dict
        """
        return {band_name: self.fir_filter(band_range[0], band_range[1], **kwargs) for band_name, band_range in EEG_BANDS.items()}

    def cheby_filter_bands(self, **kwargs) -> dict:
        """Return a dictionary of filtered EEG data. 

        :return: Dictionary of filtered data. {band_name: data}
        :rtype: dict
        """
        return {band_name: self.apply_cheby_filter(band_range[0], band_range[1], **kwargs) for band_name, band_range in EEG_BANDS.items()}

    def append(self, new_data) -> None:
        """Append new data to the EEG data.

        :param new_data: Data to append.
        :type new_data: EEGSeries
        """
        return self(data=np.append(self.data, new_data.data, axis=0))

    def __iter__(self):
        return iter(self.data)
