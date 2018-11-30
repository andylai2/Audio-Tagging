"""
model Keras Implementation

BibTeX Citation:

@inproceedings{krizhevsky2012imagenet,
  title={Imagenet classification with deep convolutional neural networks},
  author={Krizhevsky, Alex and Sutskever, Ilya and Hinton, Geoffrey E},
  booktitle={Advances in neural information processing systems},
  pages={1097--1105},
  year={2012}
}
"""

# Import necessary packages
import argparse

# Import necessary components to build LeNet
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

def alexnet(img_shape=(224, 224, 3), n_classes=10, l2_reg=0.):

	# Initialize model
	model = Sequential()

	# Layer 1
	model.add(Conv2D(96, (11, 11), input_shape=img_shape,
		padding='same', kernel_regularizer=l2(l2_reg)))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 2
	model.add(Conv2D(256, (5, 5), padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 3
	model.add(ZeroPadding2D((1, 1)))
	model.add(Conv2D(512, (3, 3), padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 4
	model.add(ZeroPadding2D((1, 1)))
	model.add(Conv2D(1024, (3, 3), padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))

	# Layer 5
	model.add(ZeroPadding2D((1, 1)))
	model.add(Conv2D(1024, (3, 3), padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 6
	model.add(Flatten())
	model.add(Dense(3072))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	# Layer 7
	model.add(Dense(4096))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	# Layer 8
	model.add(Dense(n_classes))
	model.add(BatchNormalization())
	model.add(Activation('softmax'))

	return model
	