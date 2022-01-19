
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


from DeepMethods import *
from DataActivity import *


#%%#

# run an experiment
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))

# run an experiment
def run_experiment(repeats,method,data_path, size_window):
    # load data
    #data_path = "./Final_data/UCI_Har_dataset.csv"
    # MotionSense Data Set: 100Hz, 24 participants;
    #6 activities: walking, walking upstairs, walking downstairs, 
    #sitting, standing and jogging.
    trainX,testX,trainy,testy=prepare_data(data_path)
    trainX,testX,trainy,testy=prepare_data_windows(trainX,testX,trainy,testy,size_window)
    trainX,testX=scale_data(trainX,testX)
    # repeat experiment
    scores = list()
    

    
    if (method =='cnn'):
     for r in range(repeats):
       
         model_cnn = evaluate_model_CNN(trainX, trainy, testX, testy)
        
         trained_model_cnn,cnn_history,score_cnn=compile_and_fit_model_CNN(model_cnn, trainX, trainy, testX, testy)
         
      
      
         #score_cnn = score_cnn * 100.0
         #print('>#%d: %.3f' % (r+1, score_cnn))
         scores.append(score_cnn)
     summarize_results(scores)  
    
    
    elif(method == 'lstm'):
     for r in range(repeats):
         model_lstm = evaluate_model_LSTM(trainX, trainy, testX, testy)
        
         trained_model_lstm,lstm_history,score_lstm=compile_and_fit_model_LSTM(model_lstm, trainX, trainy, testX, testy)
       
         
         scores.append(score_lstm)
        
     summarize_results(scores)
    elif(method =='wavelet'):
     for r in range(repeats):
         X_train_cwt = create_cwt_images(trainX, n_scales, rescale_size)

         X_test_cwt = create_cwt_images(testX, n_scales, rescale_size)


         input_shape = (X_train_cwt.shape[1], X_train_cwt.shape[2], X_train_cwt.shape[3])

         # create cnn model
         cnn_model_wav = build_cnn_model("relu", input_shape)
         # train cnn model
         trained_model_wav,wav_history,score_wav = compile_and_fit_model_wav(cnn_model, X_train_cwt, trainy, X_test_cwt, testy, 368, 10)
         scores.append(score_wav)
     summarize_results(scores) 
     
    elif(method =='all'):
     for r in range(repeats):
         X_train_cwt = create_cwt_images(trainX, n_scales, rescale_size)

         X_test_cwt = create_cwt_images(testX, n_scales, rescale_size)


         input_shape = (X_train_cwt.shape[1], X_train_cwt.shape[2], X_train_cwt.shape[3])

         # create cnn model
         cnn_model_wav = build_cnn_model("relu", input_shape)
         # train cnn model
         trainy= to_categorical(trainy)
         testy= to_categorical(testy)
         trained_model_wav,wav_history,score_wav = compile_and_fit_model_wav(cnn_model, X_train_cwt,
                                                                             trainy, X_test_cwt, testy, 368, 10)
         scores.append(score_wav)
     summarize_results(scores)  
     
     
     
     
     
     
     
     
     
     
     
    else:
     print("Invalid Method")
    
    
    











#%%#Exemple execution har dataset


data_path1 ="./Final_data/UCI_Har_dataset.csv"
data_path3="./Final_data/Pamap_dataset.csv"
data_path4="./Final_data/wHar_dataset.csv"

data_path5="./Final_data/Wisdm_dataset_phone.csv"
data_path6="./Final_data/Wisdm_dataset_watch.csv"
#%%#
run_experiment(1,'cnn',data_path1,128)
run_experiment(1,'lstm',data_path1,128)

run_experiment(1,'wavelet',data_path1,128)



run_experiment(1,'cnn',data_path2,128)
run_experiment(1,'lstm',data_path2,128)

run_experiment(1,'wavelet',data_path2,128)

run_experiment(1,'cnn',data_path3,128)
run_experiment(1,'lstm',data_path3,128)

run_experiment(1,'wavelet',data_path3,128)


run_experiment(1,'cnn',data_path4,128)
run_experiment(1,'lstm',data_path4,128)


run_experiment(1,'cnn',data_path4,128)
run_experiment(1,'lstm',data_path4,128)

run_experiment(1,'wavelet',data_path4,128)


run_experiment(1,'cnn',data_path5,100)
run_experiment(1,'lstm',data_path5,100)

run_experiment(1,'wavelet',data_path5,100)


run_experiment(1,'cnn',data_path6,100)
run_experiment(1,'lstm',data_path6,100)

run_experiment(1,'wavelet',data_path6,100)






