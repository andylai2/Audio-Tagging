import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torchvision
import librosa
import numpy as np
from skimage.transform import resize
from torch import Tensor
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torchvision import models
from config import MfccConfig
from freesound_dataloader import Freesound


config = MfccConfig(audio_duration=2.6, learning_rate=0.005, max_epochs=20)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def calculate_val_accuracy(valloader):
    """ Util function to calculate val set accuracy,
    both overall and per class accuracy
    Args:
        valloader (torch.utils.data.DataLoader): val set
    Returns:
        tuple: (overall accuracy, class level accuracy)
    """
    correct = 0.
    total = 0.
    predictions = []

    class_correct = list(0. for _ in range(config.n_classes))
    class_total = list(0. for _ in range(config.n_classes))

    for data in valloader:
        images, labels = data
        # images = images.cuda()
        # labels = labels.cuda()
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(list(predicted.cpu().numpy()))
        total += labels.size(0)
        correct += (predicted == labels).sum()

        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1

    class_accuracy = 100 * np.divide(class_correct, class_total)
    return 100*correct/total, class_accuracy


transform = transforms.Compose([
    transforms.Lambda(lambda x: x.astype(np.float32) / np.max(x)),  # rescale to -1 to 1
    transforms.Lambda(lambda x: librosa.feature.mfcc(x, sr=config.sampling_rate, n_mfcc=config.n_mfcc)),  # MFCC
    transforms.Lambda(lambda x: resize(x, (224, 224), anti_aliasing=True)),
    transforms.Lambda(lambda x: Tensor(x))
    ])

dataset = Freesound(transform=transform, mode="train")
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
test_dataset = Freesound(transform=transform, mode="test")

print("dataset size", len(dataset))
print("train size", train_size)
print("val size", val_size)

trainloader = DataLoader(train_dataset, batch_size=32,
                         shuffle=True, num_workers=4)

valloader = DataLoader(val_dataset, batch_size=256,
                       shuffle=False, num_workers=2)

testloader = DataLoader(test_dataset, batch_size=256,
                        shuffle=False, num_workers=2)

classes = {'Acoustic_guitar': 38, 'Applause': 37, 'Bark': 19, 'Bass_drum': 21, 'Burping_or_eructation': 28,
           'Bus': 22, 'Cello': 4, 'Chime': 20, 'Clarinet': 7, 'Computer_keyboard': 8, 'Cough': 17, 'Cowbell': 33,
           'Double_bass': 29, 'Drawer_open_or_close': 36, 'Electric_piano': 34, 'Fart': 14, 'Finger_snapping': 40,
           'Fireworks': 31, 'Flute': 16, 'Glockenspiel': 3, 'Gong': 26, 'Gunshot_or_gunfire': 6, 'Harmonica': 25,
           'Hi-hat': 0, 'Keys_jangling': 9, 'Knock': 5, 'Laughter': 12, 'Meow': 35, 'Microwave_oven': 27,
           'Oboe': 15, 'Saxophone': 1, 'Scissors': 24, 'Shatter': 30, 'Snare_drum': 10, 'Squeak': 23,
           'Tambourine': 32, 'Tearing': 13, 'Telephone': 18, 'Trumpet': 2, 'Violin_or_fiddle': 39, 'Writing': 11}
classes = dict((v, k) for k, v in classes.items())

net = models.AlexNet(num_classes=41)
# net = net.cuda()
net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=config.learning_rate, momentum=0.9)
# optimizer = optim.Adam(net.parameters(), lr=0.0001)

plt.ioff()
fig = plt.figure()
train_loss_over_epochs = []
val_accuracy_over_epochs = []

########################################################################
# Train the network
# ^^^^^^^^^^^^^^^^^^^^
########################################################################
print("Training Network...")

for epoch in range(config.max_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # inputs = inputs.cuda()
        # labels = labels.cuda()
        inputs, labels = inputs.to(device), labels.to(device)

        inputs, labels = Variable(inputs), Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 10 == 9:  # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / (i + 1)))

    # Normalizing the loss by the total number of train batches
    running_loss /= len(trainloader)
    print('[%d] loss: %.3f' %
          (epoch + 1, running_loss))

    # Scale of 0.0 to 100.0
    # Calculate validation set accuracy of the existing model
    val_accuracy, class_accuracy = calculate_val_accuracy(valloader)
    print('Accuracy of the network on the val images: %d %%' % val_accuracy)

    train_loss_over_epochs.append(running_loss)
    val_accuracy_over_epochs.append(val_accuracy)

# Plot train loss over epochs and val set accuracy over epochs
# Nothing to change here
# -------------
plt.subplot(2, 1, 1)
plt.ylabel('Train loss')
plt.plot(np.arange(config.max_epochs), train_loss_over_epochs, 'k-')
plt.title('train loss and val accuracy')
plt.xticks(np.arange(config.max_epochs, dtype=int))
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(np.arange(config.max_epochs), val_accuracy_over_epochs, 'b-')
plt.ylabel('Val accuracy')
plt.xlabel('Epochs')
plt.xticks(np.arange(config.max_epochs, dtype=int))
plt.grid(True)
plt.savefig("plot.png")
plt.close(fig)
print('Finished Training')
# -------------

########################################################################
# Run the network on test data, and create .csv file
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
########################################################################

net.eval()

total = 0
predictions = []
for data in testloader:
    images, labels = data

    # images = images.cuda()
    # labels = labels.cuda()
    images, labels = images.to(device), labels.to(device)

    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    predictions.extend(list(predicted.cpu().numpy()))
    total += labels.size(0)

test_accuracy, class_accuracy = calculate_val_accuracy(testloader)
print('Accuracy of the network on the test images: %d %%' % test_accuracy)

with open('test.csv', 'w') as csvfile:
    wr = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    wr.writerow(["id", "prediction"])
    for l_i, label in enumerate(zip(predictions, labels)):
        wr.writerow([str(l_i), classes[label]])
