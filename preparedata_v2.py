# -*- coding: utf-8 -*-
"""
@author: Group 6
"""
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

def get_and_prepare_data_mnist():
    # load data
    #(X_train, y_train), (X_test, y_test) = mnist.load_data()
    (X_train, y_train), (X_test, y_test) = mnist.load_data("/chemin/absolu/vers/mnist.npz")
    # reshape to be [samples][width][height][pixels]
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
    
    
    # normalize inputs from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255
    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]
    return (X_train, y_train), (X_test, y_test), num_classes
