import numpy as np
import librosa
import os

def extract_features(filename):
	y, sample_rate = librosa.load(filename, sr=2108)
	features = librosa.feature.melspectrogram(y=y, sr=sample_rate, n_mels=128)
	return features

root = 'audio'
filenames = os.listdir(root)
os.chdir(root)
for wav_file in filenames:
	feat = extract_features(wav_file)
	dest = '../train/' + str(os.path.splitext(wav_file)[0])
	np.save(dest, feat)