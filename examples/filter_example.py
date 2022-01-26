from biosip_tools.eeg import timeseries
import matplotlib.pyplot as plt

eeg = timeseries.EEGSeries("/home/marco/data/eeg_data/EEG_7_2_C.npy")

filt_eeg = eeg.filter(0.5, 30)
fig, axs = plt.subplots(nrows=2, figsize=(30,10))
axs[0].plot(eeg.data[0,0,:500])
axs[1].plot(filt_eeg[0,0,:500])
plt.show()

filt_bands =  eeg.filter_bands()
fig, axs = plt.subplots(nrows=len(filt_bands), figsize=(30,10))
for idx, band_name in enumerate(filt_bands):
    data = filt_bands[band_name]
    axs[idx].plot(data[0,0,:500])
    axs[idx].title.set_text(band_name)
plt.show()