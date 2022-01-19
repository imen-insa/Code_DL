# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 12:42:52 2022

@author: trabelsi
"""

from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
import numpy as np 
import pandas as pd 
from sklearn.model_selection import GroupShuffleSplit
import tensorflow.compat.v1 as tf
import pywt
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
tf.disable_v2_behavior()
tf.reset_default_graph()
from keras import backend as K
K.clear_session()

#%%#

def prepare_data(path):   
    
   with open(path, 'r') as f: data = pd.read_csv(f)
   train_idx, test_idx = next(GroupShuffleSplit(test_size=.30, 
                                                   n_splits=2, random_state = 0
                                                   ).split(data, groups=data['User']))

   train = data.iloc[train_idx]
   test = data.iloc[test_idx]  
   
   trainX=train[['Acc-x','Acc-y','Acc-z','Gyr-x', 'Gyr-y','Gyr-z']]

   testX=test[['Acc-x','Acc-y','Acc-z','Gyr-x', 'Gyr-y','Gyr-z']]

   trainlabel=train[['Activity']]

   testlabel=test[['Activity']]
   return trainX,testX,trainlabel,testlabel   
#%%#
def prepare_data_windows(trainX,testX,trainlabel,testlabel,timestep):    
    
  
   remainTrain=len(trainX)%timestep
   remainTest=len(testX)%timestep
   trainX.drop(trainX.tail(remainTrain).index,inplace=True)
   testX.drop(testX.tail(remainTest).index,inplace=True)
   trainlabel.drop(trainlabel.tail(remainTrain).index,inplace=True)
   testlabel.drop(testlabel.tail(remainTest).index,inplace=True)
  
   trainX= trainX.values.reshape((-1, timestep, trainX.shape[1]))
   testX= testX.values.reshape((-1, timestep, testX.shape[1]))
   
   
   
  
  
   
   
   list1 = []
   trainlabel=trainlabel.to_numpy()
   for i in range(0,len(trainlabel),timestep):
       list1.append(trainlabel[i])
       
   trainy = np.array(list1)


   list1 = []
   testlabel=testlabel.to_numpy()
   for i in range(0,len(testlabel),timestep):
       list1.append(testlabel[i])
       
   testy = np.array(list1)
   
   
   
   
   trainy = to_categorical(trainy)
   testy = to_categorical(testy) 
   
   return trainX,testX,trainy,testy    



#%%#
# standardize data
def scale_data(trainX, testX):
    # remove overlap
    cut = int(trainX.shape[1] / 2)
    longX = trainX[:, -cut:, :]
    # flatten windows
    longX = longX.reshape((longX.shape[0] * longX.shape[1], longX.shape[2]))
    # flatten train and test
    flatTrainX = trainX.reshape((trainX.shape[0] * trainX.shape[1], trainX.shape[2]))
    flatTestX = testX.reshape((testX.shape[0] * testX.shape[1], testX.shape[2]))
    # standardize

    s = StandardScaler()
        # fit on training data
    s.fit(longX)
        # apply to training and test data
    longX = s.transform(longX)
    flatTrainX = s.transform(flatTrainX)
    flatTestX = s.transform(flatTestX)
    # reshape
    flatTrainX = flatTrainX.reshape((trainX.shape))
    flatTestX = flatTestX.reshape((testX.shape))
    return flatTrainX, flatTestX
















