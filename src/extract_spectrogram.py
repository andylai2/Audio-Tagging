import librosa
import librosa.display
import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

mean = (0.485+0.456+0.406)/3
std = (0.229+0.224+0.225)/3
sr = 44100

def extract_log_spectrogram(fname):
    y, _ = librosa.load(fname, sr=sr)
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    log_S = librosa.amplitude_to_db(S, ref=np.max)
    return log_S

def extract_mfcc(fname):
    y, _ = librosa.load(fname, sr=sr)
    # y = y[:2*sr]
    features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return features

def normalize(x):
    x = -x/80
    x = (x-mean)/std

def display_spectogram(log_S):
    plt.figure(figsize=(12,4))
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
    plt.title('mel power spectrogram')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    root = '../audio'
    filenames = os.listdir(root)
    os.chdir(root)
    for wav_file in filenames:
        feat = extract_mfcc(wav_file)
        print(np.max(feat), np.min(feat))
        print(feat.shape)
        display_spectogram(feat)
        dest = '../train/' + str(os.path.splitext(wav_file)[0])
        # np.save(dest, feat)
