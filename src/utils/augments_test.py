import librosa  # conda install -c conda-forge librosa
import numpy as np
#from scipy.io.wavfile import write
import sounddevice  # conda install -c conda-forge python-sounddevice
import time

def add_noise(y, noise_amplitude):
    wgn = np.random.normal(0,noise_amplitude,y.size)  # White Gaussian noise
    y_noisy = y + wgn
    for i in range(y_noisy.size):
        y_noisy[i] = min(max(y_noisy[i],-1), 1) # bound y_noisy between -1 and 1
    return y_noisy

def dynamic_range_compression(y, amp_threshold, compress_ratio):
    y_compressed = np.copy(y)
    for i in range(y_compressed.size):
        if abs(y_compressed[i]) > amp_threshold:
            y_compressed[i] = np.sign(y_compressed[i]) * (amp_threshold + 
                        (abs(y_compressed[i]) - amp_threshold)*compress_ratio)
    return y_compressed


print('Testing augment functions')

# sr stands for sampling rate
y, sr = librosa.load('music.wav')  # a test .wav file
y_fast = librosa.effects.time_stretch(y, 2)  # twice as fast
y_third = librosa.effects.pitch_shift(y, sr, n_steps=4)  # pitch shift by 4 half steps
y_noisy = add_noise(y, noise_amplitude=0.1)
y_drc = dynamic_range_compression(y, amp_threshold=0.09, compress_ratio=0.5)


sounddevice.play(y, sr)
time.sleep(3) 
sounddevice.play(y_fast, sr)
time.sleep(3)
sounddevice.play(y_third, sr)
time.sleep(3)
sounddevice.play(y_noisy, sr)
time.sleep(3)
sounddevice.play(y_drc, sr)