
#%%#
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from tensorflow.keras.utils import to_categorical
from keras.layers import LSTM
from tensorflow.keras.layers import BatchNormalization
import numpy as np 
import pandas as pd 
from sklearn.model_selection import GroupShuffleSplit
import tensorflow.compat.v1 as tf

print(f"TensorFlow version: {tf.__version__}")
#import keras (high level API) wiht tensorflow as backend
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np 
import pandas as pd 
import os
import seaborn as sns
from skimage.transform import resize


import pywt
import matplotlib.pyplot as plt
tf.disable_v2_behavior()
tf.reset_default_graph()
from keras import backend as K
K.clear_session()

#from DeepMethods import evaluate_modell_CNN,evaluate_model_LSTM,
#...compile_and_fit_model_CNN,compile_and_fit_model_LSTM,create_cwt_images,build_cnn_model_wav,compile_and_fit_model_wav





#%%#      CNN Application 
def evaluate_model_CNN(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 0, 10, 32
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    return model
#%%# LSTM Application
def evaluate_model_LSTM(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 0, 15, 64
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    return model
  
 #%%#   
def compile_and_fit_model_CNN(model, X_train, y_train, X_test, y_test):
    verbose, epochs, batch_size = 0, 10, 32
    #compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    
    # define callbacks
    callbacks = [ModelCheckpoint(filepath='best_model.h5', monitor='accuracy', save_best_only=True)]
    
    # fit the model
    history = model.fit(x=X_train,
                        y=y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=(X_test, y_test))
    _, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
   
    
    return model, history,accuracy
   #%%#    
def compile_and_fit_model_LSTM(model, X_train, y_train, X_test, y_test):
     
    verbose, epochs, batch_size = 0, 15, 64
     #compile the model
    model.compile(
         optimizer='adam',
         loss='categorical_crossentropy',
         metrics=['accuracy'])
     
     # define callbacks
    callbacks = [ModelCheckpoint(filepath='best_model.h6', monitor='accuracy', save_best_only=True)]
     
     # fit the model
    history = model.fit(x=X_train,
                         y=y_train,
                         batch_size=batch_size,
                         epochs=epochs,
                         verbose=1,
                         callbacks=callbacks,
                         validation_data=(X_test, y_test))
    _, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0) 
    return model, history,accuracy

#%%#


def create_cwt_images(X, n_scales, rescale_size, wavelet_name = "morl"):
    n_samples = X.shape[0] 
    n_signals = X.shape[2] 
    
    # range of scales from 1 to n_scales
    scales = np.arange(1, n_scales + 1) 
    
    # pre allocate array
    X_cwt = np.ndarray(shape=(n_samples, rescale_size, rescale_size, n_signals), dtype = 'float32')
    
    for sample in range(n_samples):
        if sample % 1000 == 0:
            print(sample)
        for signal in range(n_signals):
            serie = X[sample, :, signal]
            # continuous wavelet transform 
            coeffs, freqs = pywt.cwt(serie, scales, wavelet_name)
            # resize the 2D cwt coeffs
            rescale_coeffs = resize(coeffs, (rescale_size, rescale_size), mode = 'constant')
            X_cwt[sample,:,:,signal] = rescale_coeffs
            
    return X_cwt
  
# amount of pixels in X and Y 
rescale_size = 64
# determine the max scale size
n_scales = 64



def build_cnn_model_wav(activation, input_shape):
    model = Sequential()
    
    # 2 Convolution layer with Max polling
    model.add(Conv2D(32, 5, activation = activation, padding = 'same', input_shape = input_shape))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, 5, activation = activation, padding = 'same', kernel_initializer = "he_normal"))
    model.add(MaxPooling2D())  
    model.add(Flatten())
    
    # 3 Full connected layer
    model.add(Dense(128, activation = activation, kernel_initializer = "he_normal"))
    model.add(Dense(54, activation = activation, kernel_initializer = "he_normal"))
    model.add(Dense(6, activation = 'softmax')) # 6 classes
    
    # summarize the model
    print(model.summary())
    return model

def compile_and_fit_model_wav(model, X_train, y_train, X_test, y_test, batch_size, n_epochs):

    # compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'])
    
    # define callbacks
    callbacks = [ModelCheckpoint(filepath='best_model.h7', monitor='val_sparse_categorical_accuracy', save_best_only=True)]
    
    # fit the model
    history = model.fit(x=X_train,
                        y=y_train,
                        batch_size=batch_size,
                        epochs=n_epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=(X_test, y_test))
    _, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0) 
    return model, history,accuracy




