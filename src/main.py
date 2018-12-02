import csv
import torch
import torchvision
import librosa
import numpy as np
from torch import Tensor
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from scipy.io import wavfile
from torchvision import transforms
from torchvision import models
from config import MfccConfig
from freesound_dataloader import Freesound
from models.network import Net


config = MfccConfig()


def train_val_split(validation_split=0.2, shuffle_dataset=True):
    """ Util function to generate indices for train/val splits
    Args:
        validation_split: validation split size (0.0-1.0)
        shuffle_dataset: whether dataset should be shuffled
    Returns:
        tuple: (training set indices, validation set indices
    """
    dataset_size = len(Freesound(mode="train"))
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(42)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    return train_indices, val_indices


def calculate_val_accuracy(valloader):
    """ Util function to calculate val set accuracy,
    both overall and per class accuracy
    Args:
        valloader (torch.utils.data.DataLoader): val set
    Returns:
        accuracy
    """
    correct = 0.
    total = 0.
    predictions = []

    for data in valloader:
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(list(predicted.cpu().numpy()))
        total += labels.size(0)
        correct += (predicted == labels).sum()

    return 100*correct/total


transform = transforms.Compose([
    lambda x: x.astype(np.float32) / np.max(x),  # rescale to -1 to 1
    lambda x: librosa.feature.mfcc(x, sr=config.sampling_rate, n_mfcc=config.n_mfcc),  # MFCC
    lambda x: Tensor(x)
    ])

dataset = Freesound(transform=transform, mode="train")
train_indices, val_indices = train_val_split()
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

trainloader = DataLoader(dataset, batch_size=32,
                         shuffle=True, num_workers=2, sampler=train_sampler)

valloader = DataLoader(dataset, batch_size=256,
                       shuffle=False, num_workers=2, sampler=val_sampler)

testloader = DataLoader(Freesound(transform=transform, mode="test"), batch_size=256,
                        shuffle=False, num_workers=2)

classes = {'Acoustic_guitar': 38, 'Applause': 37, 'Bark': 19, 'Bass_drum': 21, 'Burping_or_eructation': 28,
           'Bus': 22, 'Cello': 4, 'Chime': 20, 'Clarinet': 7, 'Computer_keyboard': 8, 'Cough': 17, 'Cowbell': 33,
           'Double_bass': 29, 'Drawer_open_or_close': 36, 'Electric_piano': 34, 'Fart': 14, 'Finger_snapping': 40,
           'Fireworks': 31, 'Flute': 16, 'Glockenspiel': 3, 'Gong': 26, 'Gunshot_or_gunfire': 6, 'Harmonica': 25,
           'Hi-hat': 0, 'Keys_jangling': 9, 'Knock': 5, 'Laughter': 12, 'Meow': 35, 'Microwave_oven': 27,
           'Oboe': 15, 'Saxophone': 1, 'Scissors': 24, 'Shatter': 30, 'Snare_drum': 10, 'Squeak': 23,
           'Tambourine': 32, 'Tearing': 13, 'Telephone': 18, 'Trumpet': 2, 'Violin_or_fiddle': 39, 'Writing': 11}
classes = dict((v, k) for k, v in classes.items())

net = models.alexnet()
net = net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=config.learning_rate, momentum=0.9)
# optimizer = optim.Adam(net.parameters(), lr=config.learning_rate)

train_loss_over_epochs = []
val_accuracy_over_epochs = []

########################################################################
# Train the network
# ^^^^^^^^^^^^^^^^^^^^
########################################################################

for epoch in config.max_epochs:
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        inputs = inputs.cuda()
        labels = labels.cuda()

        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]

    # Normalizing the loss by the total number of train batches
    running_loss /= len(trainloader)
    print('[%d] loss: %.3f' %
          (epoch + 1, running_loss))

    # Scale of 0.0 to 100.0
    # Calculate validation set accuracy of the existing model
    val_accuracy = calculate_val_accuracy(valloader)
    print('Accuracy of the network on the val images: %d %%' % (val_accuracy))

    train_loss_over_epochs.append(running_loss)
    val_accuracy_over_epochs.append(val_accuracy)

print('Finished Training')

########################################################################
# Run the network on test data, and create .csv file
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
########################################################################

net.eval()

total = 0
predictions = []
for data in testloader:
    images, labels = data

    images = images.cuda()
    labels = labels.cuda()

    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    predictions.extend(list(predicted.cpu().numpy()))
    total += labels.size(0)

with open('test.csv', 'w') as csvfile:
    wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    wr.writerow(["fname", "prediction"])
    for l_i, label in enumerate(predictions):
        wr.writerow([str(l_i), classes[label]])
