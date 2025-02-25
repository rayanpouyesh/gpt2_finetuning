pip install pydub!
pip install noisereduce!
pip install scipy!

from scipy.io import wavfile
import noisereduce as nr
# load data
rate, data = wavfile.read("s1.wav")
# perform noise reduction
reduced_noise = nr.reduce_noise(y=data, sr=rate, n_std_thresh_stationary=0.25,stationary=True)
wavfile.write("s1_reduced.wav", rate, reduced_noise)
import numpy as np
import librosa
import soundfile as sf

def calculate_snr(signal, noise):
    # Signal Power
    #signal_power = np.mean(signal  2)
    signal_power = np.mean(np.square(signal))

    # Noise Power
    noise_power = np.mean(np.square(noise))

    # SNR in dB
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

    # Load audio file (signal)
signal, sr = librosa.load('s1.wav', sr=None)
designal, desr = librosa.load('s1_reduced.wav', sr=None)

# Assuming the noise is the difference between the signal and a denoised version
# For demonstration, we'll generate synthetic noise as an example
# In practice, use a denoising method to estimate the noise component
noise = signal - designal

# Calculate SNR
snr = calculate_snr(signal, noise)
print(f"SNR: {snr:.4f} dB")
