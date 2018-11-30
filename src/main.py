import torch
import torchvision
import librosa
import numpy as np
from torch import Tensor
from torch import optim
from torch.utils.data import DataLoader
from scipy.io import wavfile
from torchvision import transforms
from torchvision import models
from src.mcc_dataloader import Freesound


tsfm = transforms.Compose([
    lambda x: x.astype(np.float32) / np.max(x),  # rescale to -1 to 1
    lambda x: librosa.feature.mfcc(x, sr=44100, n_mfcc=40),  # MFCC
    lambda x: Tensor(x)
    ])

dataloader = DataLoader(
    Freesound(transform=tsfm, mode="train"),
    batch_size=16,
    shuffle=True,
    num_workers=4)

net = models.alexnet()
