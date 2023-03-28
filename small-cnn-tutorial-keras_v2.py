# -*- coding: utf-8 -*-
"""
@author: Group 6
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D

import preparedata_v2 as pr
import cnnutils_v2 as cu

from os import system
a = system("clear")
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# import MNIST dataset
(X_train, y_train), (X_test, y_test), num_classes = pr.get_and_prepare_data_mnist()

# define the small model
def small_model():
    # create model
    model = Sequential()
    #model.add(Conv2D(64, (3, 3), input_shape=(1, 28, 28), activation='relu'))
    model.add(Conv2D(64, (3, 3), input_shape=(28, 28, 1), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# build the model
model = small_model()
# Fit the model
a = system("clear")
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)

# Evaluate the model
cu.print_model_error_rate(model, X_test, y_test)
# Save the model
cu.save_keras_model(model, "save_model_v2/small_model_v2_cnn")
