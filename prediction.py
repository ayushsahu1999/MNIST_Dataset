# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 22:50:55 2020

@author: Dell
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gzip
from image_preprocess_final import img_pre
from label_data import train_labels, test_labels

import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# initializing the CNN
classifier = Sequential()

# Convolution
# Formula for checking if the filter will properly fit the input size is ((W-F+2P)/(S+1))

# 1st Convolution
classifier.add(Convolution2D(32, 3, 3, strides=(1, 1), padding='same', input_shape=(28, 28, 1),
                             activation='relu'))

# 1st Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# 2nd Convolution
classifier.add(Convolution2D(64, 14, 14, strides=(1, 1), padding='same', input_shape=(14, 14, 32),
                             activation='relu'))

# 2nd Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Flattening
# By now the output shape should be 64*7*7
classifier.add(Flatten())

# Full Connection
classifier.add(Dense(output_dim=128, activation='relu'))
   
# Output layer
classifier.add(output_dim=10, activation='softplus')

# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting images to CNN
from keras.preprocessing.image import ImageDataGenerator

# Splitting training and test dataset
X_train = img_pre('samples/train-images-idx3-ubyte')
X_test = img_pre('samples/t10k-images-idx3-ubyte')
y_train = train_labels('compressed_mnist/train-labels-idx1-ubyte.gz')
y_test = test_labels('compressed_mnist/t10k-labels-idx1-ubyte.gz')


# Adjusting the training and testing results
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

# Image Augmentation
datagen = ImageDataGenerator(
                            rotation_range=20,
                            rescale = 1%255,
                            shear_image = 0.2,
                            zoom_range = 0.2,
                            horizontal_flip = True)

datagen.fit(X_train)

classifier.fit_generator(datagen.flow(X_train, y_train, batch_size=32),
                         steps_per_epoch=len(X_train)/32, epochs=100)

# Predicting the results
y_pred = classifier.predict(X_test)