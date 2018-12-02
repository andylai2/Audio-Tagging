import numpy as np


class MfccConfig:
    def __init__(self,
                 sampling_rate=44100, audio_duration=2, n_classes=41,
                 n_folds=10, learning_rate=0.0001, max_epochs=50, n_mfcc=40):
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.n_classes = n_classes
        self.n_mfcc = n_mfcc
        self.n_folds = n_folds
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        self.audio_length = self.sampling_rate * self.audio_duration
        self.dim = (self.n_mfcc, 1 + int(np.floor(self.audio_length/512)), 1)


class MelSpecConfig:
    def __init__(self,
                 sampling_rate=44100, audio_duration=2, n_classes=41,
                 n_folds=10, learning_rate=0.0001, max_epochs=50, n_mels=128):
        self.sampling_rate = sampling_rate
        self.audio_duration = audio_duration
        self.n_classes = n_classes
        self.n_mels = n_mels
        self.n_folds = n_folds
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        self.audio_length = self.sampling_rate * self.audio_duration
