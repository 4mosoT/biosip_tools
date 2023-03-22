import sys
sys.path.append('/home/marco/Projects/Doctorado/biosip_tools/')
from biosip_tools.eeg import utils
from biosip_tools.eeg import timeseries
    
eeg20 = timeseries.EEGSeries(path="/home/marco/data/eeg_data/EEG_7_20_C.npy", subjects_info="/home/marco/data/eeg_data/subjects_order_20_C.txt")
eeg2 = timeseries.EEGSeries(path="/home/marco/data/eeg_data/EEG_7_2_C.npy", subjects_info="/home/marco/data/eeg_data/subjects_order_2_C.txt")
eeg8 = timeseries.EEGSeries(path="/home/marco/data/eeg_data/EEG_7_8_C.npy", subjects_info="/home/marco/data/eeg_data/subjects_order_8_C.txt")

