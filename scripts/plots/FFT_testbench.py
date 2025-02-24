import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, iirnotch
from matplotlib import pyplot as plt
import scipy.signal as signal


#data_origin = f"C:\\Users\\Becky\\Documents\\UCAR_ImportantStuff\\Turkiye\\data\\station_TSMS00\\TSMS00_CompleteRecord_FINAL.csv"
data_origin = r"/Users/rzieber/Documents/3D-PAWS/Turkiye/reformatted/CSV_Format/3DPAWS/full_dataperiod/station_TSMS00/complete/TSMS00_CompleteRecord_FINAL.csv"

""" 
==================================================
TUTORIAL: https://realpython.com/python-scipy-fft/
==================================================
"""

# SAMPLE_RATE = 44100  # Hertz
# #SAMPLE_RATE = 0.0167 # 1x minute in Hz
# DURATION = 5  # Seconds

# def generate_sine_wave(freq, sample_rate, duration):
#     x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
#     frequencies = x * freq
#     # 2pi because np.sin takes radians
#     y = np.sin((2 * np.pi) * frequencies)
#     return x, y

# # Step 1: Generate a 2 hertz sine wave that lasts for 5 seconds
# # x, y = generate_sine_wave(2, SAMPLE_RATE, DURATION)
# # plt.plot(x, y)
# # plt.show()
# # -------------------------------------

# # Step 2: Generate a noise sine and a sine we want to preserve
# _, nice_tone = generate_sine_wave(400, SAMPLE_RATE, DURATION)
# _, noise_tone = generate_sine_wave(4000, SAMPLE_RATE, DURATION)
# noise_tone = noise_tone * 0.3

# mixed_tone = nice_tone + noise_tone

# normalized_tone = np.int16((mixed_tone / mixed_tone.max()) * 32767) # The next step is normalization, 
#                                                                     # or scaling the signal to fit into the target format. 
#                                                                     # Due to how you’ll store the audio later, 
#                                                                     # your target format is a 16-bit integer, 
#                                                                     # which has a range from -32768 to 32767

# # plt.plot(normalized_tone[:1000])
# # plt.show()
# # -------------------------------------


# # Step 3: Use the FFT to calculate the frequency spectrum
# # Number of samples in normalized_tone
# N = SAMPLE_RATE * DURATION  # total number of samples

# yf = fft(normalized_tone)
# xf = fftfreq(N, 1 / SAMPLE_RATE)

# plt.plot(xf, np.abs(yf))
# plt.show()
# # -------------------------------------


"""
Attempt number 2 -- plot a simple graph showing the spikes in frequency across the dataset.
"""
df = pd.read_csv(data_origin)

temperature_data = df['htu_temp'].to_numpy()

clean_data = temperature_data[                  # FFT requires noise get removed
    ~np.isnan(temperature_data) & \
        (temperature_data >= -50) & \
            (temperature_data < 50)
    ]

#np.savetxt("/Users/rzieber/Downloads/TSMS00_htuTemp_cleaned.txt", clean_data)
print("Mean:", np.mean(clean_data))
print("Standard Deviation:", np.std(clean_data))

clean_data = clean_data - np.mean(clean_data)   # remove the DC offset
                                                # This is called 'detrending' the data

# PHASEONE Exploration of the frequencies present in the data ----------------------
#Fs = 1/60  # 1 sample per minute, converted to Hz
# #Fs = 1 / (60 * 60 * 24)  # Convert to samples per day
# # n = 2**np.ceil(np.log2(len(clean_data))).astype(int)  # Next power of 2
# # padded_data = np.pad(clean_data, (0, n - len(clean_data)), mode='constant')

# # Perform FFT
# n = len(clean_data)
# fft_result = fft(clean_data)
# # n = len(padded_data)
# # fft_result = fft(padded_data)
# fft_result_normalized = fft_result / n
# frequencies = np.fft.fftfreq(n, 1/Fs)

# #np.savetxt("C:\\Users\\Becky\\Downloads\\TSMS00_Frequencies.txt", frequencies)

# # Plot the magnitude spectrum
# plt.figure(figsize=(10, 6))
# plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_result[:len(frequencies)//2]))
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Magnitude')
# plt.title('TSMS00: Frequency Spectrum of Temperature Data [DETRENDED]')
# plt.grid(True)
# plt.show()
# plt.close()
# -------------------------------------------------------------------------------------

fs = 0.0167 # Sampling Frequency Hz

# Design a notch filter to remove specific frequency noise
notch_freq = 0.00834  # Frequency to be removed (Hz)
quality_factor = 10  # Quality factor (higher Q = narrower notch)

# Design the notch filter coefficients
b, a = iirnotch(notch_freq, quality_factor, fs)

# Apply the notch filter
filtered_signal = filtfilt(b, a, clean_data)

# Save or analyze the filtered signal
print("Filtered signal mean:", np.mean(filtered_signal))
print("Filtered signal standard deviation:", np.std(filtered_signal))

time_clean = np.arange(len(clean_data)) * (1 / fs)  # Time in seconds
time_filtered = np.arange(len(filtered_signal)) * (1 / fs)  # Time in seconds

# Plot the original and filtered signals
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time_clean, clean_data, label='Original Signal')
plt.title('Original Signal with Noise')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(time_filtered, filtered_signal, label='Filtered Signal', color='orange')
plt.title('Filtered Signal (Notch Filter Applied)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


