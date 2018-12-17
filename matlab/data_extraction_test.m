% Test script to play with data 

datadir = '.\data';
testdatadir = [datadir, filesep,'audio_test'];
traindatadir = [datadir, filesep,'audio_train'];

% initialize image directory if it doesn't exist already
imagedir = [datadir, filesep, 'images_train'];
spectdir = fullfile(imagedir, 'spect');
mfccdir = fullfile(imagedir, 'mfcc');
crpdir = fullfile(imagedir, 'crp');
if (~exist(imagedir, 'dir'))
    mkdir(imagedir);
    mkdir(spectdir);
    mkdir(mfccdir);
    mkdir(crpdir);
end

train_names = dir(traindatadir);
train_names = train_names(3:end); %throw out ., ..
%N = size(train_names,1);
N = 10; % practice on 10 elements
fs = 44100; % sampling rate
fivesec = 5 * fs; % five seconds worth of samples
audio_train = zeros(fivesec,N);
thresh_volume = 0.01; 

%% Read data from file
for i = 1:N % discount the first two names
    temp_data = audioread([traindatadir, filesep, train_names(i).name]);
    l = size(temp_data,1);
    % take the first five seconds of sound
    % if less than 5 seconds of sound, just take five seconds
    % otherwise, zero pad up to 5 seconds
    if l < fivesec
        audio_train(:,i) = padarray(temp_data,fivesec-l,0,'post');
    else
        k = find(temp_data > thresh_volume, 1, 'first');
        if l - k < fivesec
            audio_train(:,i) = temp_data(l-fivesec+1:len);
        else 
            audio_train(:,i) = temp_data(k:k+fivesec-1);
        end
    end
end

%% Perform time dilation
audio_train_06 = resample(audio_train,3,5);
audio_train_075 = resample(audio_train,3,4);
audio_train_09 = resample(audio_train,9,10);
audio_train_11 = resample(audio_train,11,10);
audio_train_125 = resample(audio_train,5,4);
audio_train_14 = resample(audio_train,7,4);

%% Generate images and store to imagedir
for i = 1:N
    % Generate spectrograms with a .0125s window 
    % so it's roughly square
    nwin = floor(.0125 * fs);
    nov = floor( nwin/4 );
    % original
    s = abs( spectrogram( audio_train(:,i), hamming( nwin ), nov ) );
    imname = ['spect_', train_names(i).name(1:end-4), '.png'];
    imwrite(s, fullfile(spectdir, imname));
    % 0.6
    s = abs( spectrogram( audio_train_06(:,i), hamming( nwin ), nov ) );
    imname = ['spect_', train_names(i).name(1:end-4), '_06.png'];
    imwrite(s, fullfile(spectdir, imname));
    % 0.75
    s = abs( spectrogram( audio_train_075(:,i), hamming( nwin ), nov ) );
    imname = ['spect_', train_names(i).name(1:end-4), '_075.png'];
    % 0.9
    imwrite(s, fullfile(spectdir, imname));
    s = abs( spectrogram( audio_train_09(:,i), hamming( nwin ), nov ) );
    imname = ['spect_', train_names(i).name(1:end-4), '_09.png'];
    imwrite(s, fullfile(spectdir, imname));
    % 1.1
    s = abs( spectrogram( audio_train_11(:,i), hamming( nwin ), nov ) );
    imname = ['spect_', train_names(i).name(1:end-4), '_11.png'];
    imwrite(s, fullfile(spectdir, imname));
    % 1.25
    s = abs( spectrogram( audio_train_125(:,i), hamming( nwin ), nov ) );
    imname = ['spect_', train_names(i).name(1:end-4), '_125.png'];
    imwrite(s, fullfile(spectdir, imname));
    % 1.4
    s = abs( spectrogram( audio_train_14(:,i), hamming( nwin ), nov ) );
    imname = ['spect_', train_names(i).name(1:end-4), '_14.png'];
    imwrite(s, fullfile(spectdir, imname));
    
    % Generate MFCC image with 50 cepstral coefficients
    % original
    c = abs( mfcc( audio_train(:,i), 50, 'Nw', nwin, ...
        'No', nov, 'Fs', fs, 'M', 50 ) );
    imname = ['mfcc_', train_names(i).name(1:end-4), '.png'];
    imwrite(c, fullfill(mfcc, imname));
    c = abs( mfcc( audio_train_06(:,i), 50, 'Nw', nwin, ...
        'No', nov, 'Fs', fs, 'M', 50 ) );
    imname = ['mfcc_', train_names(i).name(1:end-4), '_06.png'];
    imwrite(c, fullfill(mfcc, imname));
    c = abs( mfcc( audio_train_075(:,i), 50, 'Nw', nwin, ...
        'No', nov, 'Fs', fs, 'M', 50 ) );
    imname = ['mfcc_', train_names(i).name(1:end-4), '_075.png'];
    imwrite(c, fullfill(mfcc, imname));
    c = abs( mfcc( audio_train_09(:,i), 50, 'Nw', nwin, ...
        'No', nov, 'Fs', fs, 'M', 50 ) );
    imname = ['mfcc_', train_names(i).name(1:end-4), '_09.png'];
    imwrite(c, fullfill(mfcc, imname));
    c = abs( mfcc( audio_train_11(:,i), 50, 'Nw', nwin, ...
        'No', nov, 'Fs', fs, 'M', 50 ) );
    imname = ['mfcc_', train_names(i).name(1:end-4), '_11.png'];
    imwrite(c, fullfill(mfcc, imname));
    c = abs( mfcc( audio_train_125(:,i), 50, 'Nw', nwin, ...
        'No', nov, 'Fs', fs, 'M', 50 ) );
    imname = ['mfcc_', train_names(i).name(1:end-4), '_125.png'];
    imwrite(c, fullfill(mfcc, imname));
    c = abs( mfcc( audio_train_14(:,i), 50, 'Nw', nwin, ...
        'No', nov, 'Fs', fs, 'M', 50 ) );
    imname = ['mfcc_', train_names(i).name(1:end-4), '_14.png'];
    imwrite(c, fullfill(mfcc, imname));
    
    % Generate crp image
end