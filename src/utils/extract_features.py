import sys, os
import numpy as np
import librosa
import scipy
from scipy.io import wavfile
from scipy.signal import spectrogram
from librosa.feature import mfcc
import csv
from librosa.core import resample
from skimage.transform import resize
import PIL

def clip_audio(x, T_clip, thresh_amp):
    # clips or zero pads audio file to T_clip length
    l = x.shape[0]
    if l < T_clip:
        y = np.hstack( ( x,np.zeros(T_clip-l) ) )
    else:
        k = np.argwhere(x >= thresh_amp)[0]
        if l - k < T_clip:
            y = x[l-T_clip:l]
        else:
            y = x[np.arange(k,k+T_clip)]
    return y

def square_spect(x,f):
    N_win = 524
    N_ov = N_win/4
    win = scipy.signal.get_window('hamming',N_win)
    _f,_t,Sxx = spectrogram(x, f, window=win, nperseg = N_win, noverlap=N_ov)
    Sxx = resize(Sxx,(224,224))
    return Sxx
    #return Sxx.astype(uint32)

def square_mfcc(x,f):
    cc = librosa.feature.mfcc(x, f,n_mfcc = 80)
    cc = 20*np.log10(resize(cc, (244,244)))
    return cc.astype(float32)

currDir = os.getcwd()
dataDir = os.path.join(currDir, 'data')
testDataDir = os.path.join(dataDir, 'audio_test')
trainDataDir = os.path.join(dataDir, 'audio_train')

dataInfoFile = os.path.join(dataDir, 'train.csv')

# initialize image directories if nonexistent
imageDir = os.path.join(dataDir, 'images_train')
spectDir = os.path.join(imageDir, 'spect')
mfccDir = os.path.join(imageDir, 'mfcc')
#crpDir = os.path.join(imageDir, 'crp')
if not os.path.isdir(imageDir):
    os.mkdir(imageDir)
    os.mkdir(spectDir)
    os.mkdir(mfccDir)
    #os.mkdir(crpDir)

# read in the wavfiles of just the desired classes

# TODO: set up classes as a dict?
classes = ['Applause','Bark','Bus','Computer_keyboard','Cough',
           'Drawer_open_or_close','Fireworks','Gunshot_or_gunfire',
           'Keys_jangling','Laughter','Shatter','Tearing','Writing']

dataNames = []
with open(dataInfoFile, 'r',newline='') as infile:
    #has_header = csv.Sniffer().has_header(file.read(1024))
    #file.seek(0)
    reader = csv.reader(infile, delimiter=',')
    #if has_header:
    #    next(reader)
    for line in reader:
        if line[1] in classes:
            dataNames.append(line[0])
infile.close()

N_data = len(dataNames)
fs = 44100 # sampling rater
fs_dilate = np.floor( fs * np.asarray([1, .6, .75, .9, .11, .125, .14]))
T_clip = 2 * fs # two seconds worth of samples
thresh_amp = 1500; # level at which we consider there to be sound
#audio_train_1 = np.zeros((T_clip,N_data)) # original data matrix
#audio_train_06 = np.zeros_like(audio_train_1) # time delated (and below)
#audio_train_075 = np.zeros_like(audio_train_1)
#audio_train_09 = np.zeros_like(audio_train_1)
#audio_train_11 = np.zeros_like(audio_train_1)
#audio_train_125 = np.zeros_like(audio_train_1)
#audio_train_14 = np.zeros_like(audio_train_1)

for i in range(N_data):
#for i in range(10):   
    f, x_1 = wavfile.read(os.path.join(trainDataDir, dataNames[i]))
    x_1 = x_1.astype(float)
    x_06 = resample(x_1, f, fs_dilate[1])
    x_075 = resample(x_1, f, fs_dilate[2])
    x_09 = resample(x_1, f, fs_dilate[3])
    x_11 = resample(x_1, f, fs_dilate[4])
    x_125 = resample(x_1, f, fs_dilate[5])
    x_14 = resample(x_1, f, fs_dilate[6])
    
    # clip each audio file to two seconds
    y_1 = clip_audio(x_1, T_clip, thresh_amp)
    y_06 = clip_audio(x_06, T_clip, thresh_amp)
    y_075 = clip_audio(x_075, T_clip, thresh_amp)
    y_09 = clip_audio(x_09, T_clip, thresh_amp)
    y_11 = clip_audio(x_11, T_clip, thresh_amp)
    y_125 = clip_audio(x_125, T_clip, thresh_amp)
    y_14 = clip_audio(x_14, T_clip, thresh_amp)
    
    # generate spectrograms of size 224x224
    s_1 = square_spect(y_1,f)
    image_path = os.path.join(spectDir, 'sp_' + dataNames[i][:-4] + '_1.png')
    PIL.Image.fromarray(s_1).convert('L').save(image_path)
    s_1 = square_spect(y_06,f)
    image_path = os.path.join(spectDir, 'sp_' + dataNames[i][:-4] + '_06.png')
    PIL.Image.fromarray(s_1).convert('L').save(image_path)
    s_1 = square_spect(y_075,f)
    image_path = os.path.join(spectDir, 'sp_' + dataNames[i][:-4] + '_075.png')
    PIL.Image.fromarray(s_1).convert('L').save(image_path)
    s_1 = square_spect(y_09,f)
    image_path = os.path.join(spectDir, 'sp_' + dataNames[i][:-4] + '_09.png')
    PIL.Image.fromarray(s_1).convert('L').save(image_path)
    s_1 = square_spect(y_11,f)
    image_path = os.path.join(spectDir, 'sp_' + dataNames[i][:-4] + '_11.png')
    PIL.Image.fromarray(s_1).convert('L').save(image_path)
    s_1 = square_spect(y_125,f)
    image_path = os.path.join(spectDir, 'sp_' + dataNames[i][:-4] + '_125.png')
    PIL.Image.fromarray(s_1).convert('L').save(image_path)
    s_1 = square_spect(y_14,f)
    image_path = os.path.join(spectDir, 'sp_' + dataNames[i][:-4] + '_14.png')
    PIL.Image.fromarray(s_1).convert('L').save(image_path)
   
    # generate square mel spectrograms of size 224x224
    
    cc = square_mfcc(y_1,f)
    image_path = os.path.join(mfccDir, 'mfcc_' + dataNames[i][:-4] + '_1.png')
    PIL.Image.fromarray(cc).convert('L').save(image_path)
    cc = square_mfcc(y_06,f)
    image_path = os.path.join(mfccDir, 'mfcc_' + dataNames[i][:-4] + '_06.png')
    PIL.Image.fromarray(cc).convert('L').save(image_path)
    cc = square_mfcc(y_075,f)
    image_path = os.path.join(mfccDir, 'mfcc_' + dataNames[i][:-4] + '_075.png')
    PIL.Image.fromarray(cc).convert('L').save(image_path)
    cc = square_mfcc(y_09,f)
    image_path = os.path.join(mfccDir, 'mfcc_' + dataNames[i][:-4] + '_09.png')
    PIL.Image.fromarray(cc).convert('L').save(image_path)
    cc = square_mfcc(y_11,f)
    image_path = os.path.join(mfccDir, 'mfcc_' + dataNames[i][:-4] + '_11.png')
    PIL.Image.fromarray(cc).convert('L').save(image_path)
    cc = square_mfcc(y_125,f)
    image_path = os.path.join(mfccDir, 'mfcc_' + dataNames[i][:-4] + '_125.png')
    PIL.Image.fromarray(cc).convert('L').save(image_path)
    cc = square_mfcc(y_4,f)
    image_path = os.path.join(mfccDir, 'mfcc_' + dataNames[i][:-4] + '_4.png')
    PIL.Image.fromarray(cc).convert('L').save(image_path)
