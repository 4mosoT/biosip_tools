import numpy as np
import mne
from scipy import signal
import matplotlib.pyplot as plt
from .constants import EEG_BANDS

class EEGSeries():

    def __init__(self, data: np.ndarray = None, path: str = None, sample_rate: int = 500) -> None:
        """Class for EEG time series.

        :param data: EEG data.
        :type data: np.ndarray
        :param path: Path to .npy array. Expected shape is (n_subjects, n_channels, n_samples)
        :type path: str
        :param sample_rate: sample rate, defaults to 500
        :type sample_rate: int, optional
        """
        assert(data is not None or path is not None)
        self.data = np.load(path) if path is not None else data
        assert(len(self.data.shape) == 3)
        self.sample_rate = sample_rate

    def apply_cheby_filter(self, lowcut: float, highcut: float, order: int=6, rs: float=40, plot_response=False):
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

    def fir_filter(self, l_freq: float, h_freq: float,  verbose=False, **kwargs) -> np.ndarray:
        """Apply a FIR filter to the EEG data. Accepts arguments for mne.filter.filter_data.

        :param l_freq: Lower pass-band edge.
        :type l_freq: float
        :param h_freq: Upper pass-band edge.
        :type h_freq: float
        :return: Filtered data
        :rtype: EEGSeries
        """
        return EEGSeries(data=mne.filter.filter_data(self.data, self.sample_rate, l_freq, h_freq, verbose=verbose, **kwargs))

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
        return EEGSeries(data = np.append(self.data, new_data.data, axis=0))

    def __iter__(self):
        return iter(self.data)


