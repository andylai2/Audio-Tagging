import sys, os
import torch
import librosa
import numpy as np
import pandas as pd
from torch import Tensor
from scipy.io import wavfile
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from src.config import MfccConfig


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
            self.csv_file = pd.read_csv(os.path.join(data_root, "sample_submission.csv"))

        # dict for mapping class names into indices.
        self.classes = {cls_name: i for i, cls_name in enumerate(self.csv_file["label"].unique())}
        self.transform = transform
        self.config = config
        
    def __len__(self):
        return self.csv_file.shape[0] 

    def __getitem__(self, idx):
        filename = self.csv_file["fname"][idx]

        input_length = self.config.audio_length
        rate, data = wavfile.read(os.path.join(self.data_dir, filename))

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

        if self.mode is "train":
            label = self.classes[self.csv_file["label"][idx]]
            return data, label

        elif self.mode is "test":
            return data


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    tsfm = transforms.Compose([
        lambda x: x.astype(np.float32) / np.max(x), # rescale to -1 to 1
        lambda x: librosa.feature.mfcc(x, sr=44100, n_mfcc=40), # MFCC 
        lambda x: Tensor(x)
        ])

    # todo: multiprocessing, padding data
    dataloader = DataLoader(
        Freesound(transform=tsfm, mode="train"), 
        batch_size=1,
        shuffle=True, 
        num_workers=0)

    for index, (data, label) in enumerate(dataloader):
        print(label.numpy())
        print(data.shape)
        plt.imshow(data.numpy()[0, :, :])
        plt.show()

        if index == 0:
            break
