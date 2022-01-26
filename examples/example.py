from biosip_tools.eeg import timeseries
import matplotlib.pyplot as plt

data = timeseries.EEGSeries("/home/marco/data/eeg_data/EEG_7_2_C.npy")

bands =  data.filter_eeg(0.5, 30)

fig, axs = plt.subplots(nrows=len(EEG_BANDS), figsize=(30,10))
for idx, x in enumerate(bands):
    axs[idx].plot(x[1][:500])
    axs[idx].title.set_text(x[0])

plt.show()
