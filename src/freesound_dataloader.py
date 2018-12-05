import sys, os
import torch
import librosa
import numpy as np
import pandas as pd
from torch import Tensor
from scipy.io import wavfile
from skimage.transform import resize
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from config import MfccConfig


class Freesound(Dataset):
    def __init__(self, config=MfccConfig(), transform=None, mode="train"):
        # setting directories for data
        data_root = "../input"
        self.mode = mode
        if self.mode is "train":
            self.data_dir = os.path.join(data_root, "audio_train")
            self.csv_file = pd.read_csv(os.path.join(data_root, "train.csv"))
        elif self.mode is "test":
            self.data_dir = os.path.join(data_root, "audio_test")
            self.csv_file = pd.read_csv(os.path.join(data_root, "test_post_competition.csv"))
            # ignore rows with label = 'None'
            self.csv_file = self.csv_file.loc[self.csv_file['label'] != 'None'].reset_index(drop=True)

        # dict for mapping class names into indices.
        # self.classes = {cls_name: i for i, cls_name in enumerate(self.csv_file["label"].unique())}
        self.classes = {'Acoustic_guitar': 38, 'Applause': 37, 'Bark': 19, 'Bass_drum': 21, 'Burping_or_eructation': 28, 'Bus': 22, 'Cello': 4, 'Chime': 20, 'Clarinet': 7,'Computer_keyboard': 8, 'Cough': 17, 'Cowbell': 33, 'Double_bass': 29, 'Drawer_open_or_close': 36, 'Electric_piano': 34, 'Fart': 14, 'Finger_snapping': 40, 'Fireworks': 31, 'Flute': 16, 'Glockenspiel': 3, 'Gong': 26, 'Gunshot_or_gunfire': 6, 'Harmonica': 25, 'Hi-hat': 0, 'Keys_jangling': 9, 'Knock': 5, 'Laughter': 12, 'Meow': 35, 'Microwave_oven': 27, 'Oboe': 15, 'Saxophone': 1, 'Scissors': 24, 'Shatter': 30, 'Snare_drum': 10, 'Squeak': 23, 'Tambourine': 32, 'Tearing': 13, 'Telephone': 18, 'Trumpet': 2, 'Violin_or_fiddle': 39,  'Writing': 11}
        self.transform = transform
        self.config = config
        
    def __len__(self):
        return self.csv_file.shape[0] 

    def __getitem__(self, idx):
        filename = self.csv_file["fname"][idx]

        input_length = self.config.audio_length
        rate, data = wavfile.read(os.path.join(self.data_dir, filename))
        # remove silence
        data, _ = librosa.effects.trim(data.astype(np.float32))

        # Random offset / Padding
        if len(data) > input_length:
            max_offset = len(data) - input_length
            offset = np.random.randint(max_offset)
            data = data[offset:(input_length + offset)]
        else:
            if input_length > len(data):
                max_offset = input_length - len(data)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

        if self.transform is not None:
            data = self.transform(data)

        data = torch.stack([data] * 3)
        label = self.classes[self.csv_file["label"][idx]]
        return data, label


if __name__ == '__main__':
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    tsfm = transforms.Compose([
        transforms.Lambda(lambda x: x.astype(np.float32) / np.max(x)), # rescale to -1 to 1
        transforms.Lambda(lambda x: librosa.feature.mfcc(x, sr=44100, n_mfcc=40)), # MFCC
        transforms.Lambda(lambda x: resize(x, (224, 224), anti_aliasing=True)),
        transforms.Lambda(lambda x: Tensor(x))
        ])

    # todo: multiprocessing, padding data
    dataloader = DataLoader(
        Freesound(transform=tsfm, mode="train"),
        batch_size=1,
        shuffle=True,
        num_workers=0)

    for index, (data, label) in enumerate(dataloader):
        print(index, label.numpy())
        print(data.shape)
        plt.imshow(data.numpy()[0, 0, :, :])
        plt.show()

        if index == 0:
            break
