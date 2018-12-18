# Using Convolutional Neural Networks for Environmental Audio Recognition

## Data 

Before running, data should be retrieved from the following link: https://www.kaggle.com/c/freesound-audio-tagging/data
Unzip data to a folder in the root directory called ./input

## Generating feature vectors 
Generate feature vectors (spectrogram, MFCC) without directly loading them into the neural networks. 
This can be done to visualize the inputs, or to simply speed up the training process without performing frequency domain calculations every time.
Run extract_features.py located in the ./src/utils. It will generate a directory where the input images will be stored. 
NOTE: implementation of this was not build into main.py

### Required libraries:
* librosa
* PIL
* scipy

## Classification without visualization
Run main.py in ./src. The main function has CUDA support and will attempt to run on GPU. 

### Required libraries:
* above
* PyTorch
* skimage

## Classification with layer visualization
Run main_visualize.py in ./src/visualize. CUDA support not build in yet. 

## Matlab
Matlab code has been deprecated.
