import librosa
import librosa.display
import os
from scipy.io import wavfile
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

mean = (0.485+0.456+0.406)/3
std = (0.229+0.224+0.225)/3
sr = 44100


def plot_wave(fname):
    y, sr = librosa.load(fname, duration=5)
    plt.figure(figsize=(12,4))
    librosa.display.waveplot(y, sr=sr)
    plt.title('Waveform')

def extract_log_spectrogram(fname):
    y, _ = librosa.load(fname, sr=sr)
    y = y[:5*sr]
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    log_S = librosa.amplitude_to_db(S, ref=np.max)
    return log_S

def extract_mfcc(fname):
    y, _ = librosa.load(fname, sr=sr)
    y = y[:5*sr]
    features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return features

def display_spectogram(log_S):
    plt.figure(figsize=(12,4))
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
    plt.title('mel power spectrogram')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()
    plt.show()

def display_mfcc(S):
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='mel')
    plt.title('mfcc')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    root = '../audio'
    filenames = os.listdir(root)
    os.chdir(root)
    for wav_file in filenames:
        plot_wave(wav_file)
        mfcc = extract_mfcc(wav_file)
        log_S = extract_log_spectrogram(wav_file)
        display_mfcc(mfcc)
        display_spectogram(log_S)
        dest = '../train/' + str(os.path.splitext(wav_file)[0])
        # np.save(dest, feat)
