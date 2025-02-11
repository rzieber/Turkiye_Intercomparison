import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
from matplotlib import pyplot as plt


data_origin = f"C:\\Users\\Becky\\Documents\\UCAR_ImportantStuff\\Turkiye\\data\\station_TSMS00\\TSMS00_CompleteRecord_FINAL.csv"


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

df = pd.read_csv(data_origin)
temperature_data = df['htu_temp'].to_numpy()

clean_data = temperature_data[
    ~np.isnan(temperature_data) & \
        (temperature_data != -999.99) & \
            ~np.isinf(temperature_data)
    ]

print(clean_data)
print("Mean:", np.mean(clean_data))
print("Standard Deviation:", np.std(clean_data))
# print(temperature_data)
# print("Mean:", np.mean(temperature_data))
# print("Standard Deviation:", np.std(temperature_data))

clean_data = clean_data - np.mean(clean_data) # remove the DC offset
#temperature_data = temperature_data - np.mean(temperature_data) 


# Assuming your temperature data is in a list or numpy array called 'temperature_data'
# Sampling rate: 1 sample per minute
Fs = 1/60  # 1 sample per minute, converted to Hz

# Perform FFT
n = len(temperature_data)
fft_result = fft(temperature_data)
fft_result_normalized = fft_result / n
frequencies = np.fft.fftfreq(n, 1/Fs)

#np.savetxt("C:\\Users\\Becky\\Downloads\\TSMS00_Frequencies.txt", frequencies)

# Plot the magnitude spectrum
plt.figure(figsize=(10, 6))
plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_result[:len(frequencies)//2]))
#plt.plot(frequencies[:n//2], np.abs(fft_result[:n//2]))
#plt.plot(frequencies[:n//2], np.abs(fft_result_normalized[:n//2]))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Frequency Spectrum of Temperature Data')
plt.grid(True)
plt.show()

