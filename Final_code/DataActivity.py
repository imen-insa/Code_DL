# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 12:42:52 2022

@author: trabelsi
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 12:42:52 2022

@author: trabelsi
"""
#%%#
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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
tf.disable_v2_behavior()
tf.reset_default_graph()
from keras import backend as K
K.clear_session()

#%%#

def prepare_data(path):   
    
   with open(path, 'r') as f: data = pd.read_csv(f)
   index_with_nan = data.index[data.isnull().any(axis=1)]
   data.drop(index_with_nan,0, inplace=True)
   
   train_idx, test_idx = next(GroupShuffleSplit(test_size=.30, 
                                                   n_splits=2, random_state = 0
                                                   ).split(data, groups=data['User']))

   train = data.iloc[train_idx]
   test = data.iloc[test_idx]  
   
   trainX=train[['Acc-x','Acc-y','Acc-z','Gyr-x', 'Gyr-y','Gyr-z']]

   testX=test[['Acc-x','Acc-y','Acc-z','Gyr-x', 'Gyr-y','Gyr-z']]

   trainlabel=train[['Label_act']]
  

   testlabel=test[['Label_act']]
   # datasetlabel gives information about the label of the dataset
   
   traindatasetlabel=train[['Dataset']]
   testdatasetlabel=test[['Dataset']]

    


   return trainX,testX,trainlabel,testlabel,traindatasetlabel,testdatasetlabel  

#%%#

def prepare_data_without_split(path):   
    
   with open(path, 'r') as f: data = pd.read_csv(f)
   index_with_nan = data.index[data.isnull().any(axis=1)]
   data.drop(index_with_nan,0, inplace=True)
   
   trainX=data[['Acc-x','Acc-y','Acc-z','Gyr-x', 'Gyr-y','Gyr-z']]


   trainlabel=data[['Activity', 'Categories']]
   
   #traindatasetlabel=data[['Dataset']]

   #return trainX,trainlabel,traindatasetlabel
   return trainX,trainlabel

#%%#

def prepare_data_user(path):   
    
   with open(path, 'r') as f: data = pd.read_csv(f)
   index_with_nan = data.index[data.isnull().any(axis=1)]
   data.drop(index_with_nan,0, inplace=True)
   
   
   
   labelUser=data[['User' ]]
   
   K= np.random.choice(np.unique(labelUser), 1, replace=False)
   
   for cls in K:
       new_data = data[data['User'] ==cls]
   
   train=new_data[['Acc-x','Acc-y','Acc-z','Gyr-x', 'Gyr-y','Gyr-z']]
   
  
   
   label=new_data[['User','Label_act','Dataset' ]]
   
   trainX,trainy = prepare_data_windows_XX(train,label[['Label_act']],50)
   trainX, testX, trainy, testy = train_test_split(trainX, trainy, test_size=0.3, random_state=42)
   
   
   # datasetlabel gives information about the label of the dataset
   
    


   return trainX,testX,trainy,testy

#%%#
def prepare_data_windows_user(trainX,testX,trainlabel,testlabel,traindatasetlabel,testdatasetlabel,trainUser,testUser,timestep):    
    
  
   remainTrain=len(trainX)%timestep
   remainTest=len(testX)%timestep
   
   if remainTrain>0:
       trainX= trainX[:-remainTrain]
       trainlabel= trainlabel[:-remainTrain]
       traindatasetlabel=traindatasetlabel[:-remainTrain]
       trainUser=trainUser[:-remainTrain]
   
   if remainTest>0:
       
      testX= testX[:-remainTest]
      testlabel= testlabel[:-remainTest]
      testdatasetlabel=testdatasetlabel[:-remainTest]
      testUser=testUser[:-remainTest]


   trainX= trainX.values.reshape((-1, timestep, trainX.shape[1]))
   testX= testX.values.reshape((-1, timestep, testX.shape[1]))
   #traindatasetlabel=traindatasetlabel.values.reshape((-1, timestep, trainX.shape[1]))
   #testdatasetlabel=testdatasetlabel.values.reshape((-1, timestep, testX.shape[1]))
   list1 = []
   trainlabel=trainlabel.to_numpy()
   for i in range(0,len(trainlabel),timestep):
       list1.append(trainlabel[i])
       
   trainy = np.array(list1)
   
   
   listUser=[]
   
   trainUser=trainUser.to_numpy()
   for i in range(0,len(trainUser),timestep):
       listUser.append(trainUser[i])
       
   trainUser = np.array(listUser)
   
   listUser=[]
   
   testUser=testUser.to_numpy()
   for i in range(0,len(testUser),timestep):
       listUser.append(testUser[i])
       
   testUser = np.array(listUser)
   
   
   
   
   
   
   
   listDtr = []
   traindatasetlabel=traindatasetlabel.to_numpy()
   for i in range(0,len(traindatasetlabel),timestep):
       listDtr.append(traindatasetlabel[i])
       
   traindatasetlabel = np.array(listDtr)
   
   listDte = []
   testdatasetlabel=testdatasetlabel.to_numpy()
   for i in range(0,len(testdatasetlabel),timestep):
       listDte.append(testdatasetlabel[i])
       
   testdatasetlabel = np.array(listDte)
   
   
   
   
   
   list2 = []
   testlabel=testlabel.to_numpy()
   for i in range(0,len(testlabel),timestep):
       list2.append(testlabel[i])
       
   testy = np.array(list2)
   
   
   trainy = to_categorical(trainy)
   testy = to_categorical(testy) 
   
   return trainX,testX,trainy,testy,traindatasetlabel,testdatasetlabel,trainUser,testUser

#%%#
def prepare_data_windows_XX(trainX,trainlabel,timestep):    
    
  
   remainTrain=len(trainX)%timestep
   
   if remainTrain>0:
       trainX= trainX[:-remainTrain]
       trainlabel= trainlabel[:-remainTrain]
     
   
   trainX= trainX.values.reshape((-1, timestep, trainX.shape[1]))
   list1 = []
   trainlabel=trainlabel.to_numpy()
   for i in range(0,len(trainlabel),timestep):
       list1.append(trainlabel[i])
       
   trainy = np.array(list1)
   
   
   trainy = to_categorical(trainy)
   
   return trainX,trainy


   if remainTrain>0:
       trainX= trainX[:-remainTrain]
       trainlabel= trainlabel[:-remainTrain]
     
   
 
   


   trainX= trainX.values.reshape((-1, timestep, trainX.shape[1]))

   #traindatasetlabel=traindatasetlabel.values.reshape((-1, timestep, trainX.shape[1]))
   #testdatasetlabel=testdatasetlabel.values.reshape((-1, timestep, testX.shape[1]))
   list1 = []
   trainlabel=trainlabel.to_numpy()
   for i in range(0,len(trainlabel),timestep):
       list1.append(trainlabel[i])
       
   trainy = np.array(list1)
  
   
   
   trainy = to_categorical(trainy)

   return trainX,trainy



#%%#
def prepare_data_windows1(trainX,testX,trainlabel,testlabel,traindatasetlabel,testdatasetlabel,timestep):    
    
  
   remainTrain=len(trainX)%timestep
   remainTest=len(testX)%timestep
   
   if remainTrain>0:
       trainX= trainX[:-remainTrain]
       trainlabel= trainlabel[:-remainTrain]
       traindatasetlabel=traindatasetlabel[:-remainTrain]
   
   if remainTest>0:
       
      testX= testX[:-remainTest]
      testlabel= testlabel[:-remainTest]
      testdatasetlabel=testdatasetlabel[:-remainTest]


   trainX= trainX.values.reshape((-1, timestep, trainX.shape[1]))
   testX= testX.values.reshape((-1, timestep, testX.shape[1]))
   #traindatasetlabel=traindatasetlabel.values.reshape((-1, timestep, trainX.shape[1]))
   #testdatasetlabel=testdatasetlabel.values.reshape((-1, timestep, testX.shape[1]))
   list1 = []
   trainlabel=trainlabel.to_numpy()
   for i in range(0,len(trainlabel),timestep):
       list1.append(trainlabel[i])
       
   trainy = np.array(list1)
   
   
   listDtr = []
   traindatasetlabel=traindatasetlabel.to_numpy()
   for i in range(0,len(traindatasetlabel),timestep):
       listDtr.append(traindatasetlabel[i])
       
   traindatasetlabel = np.array(listDtr)
   
   listDte = []
   testdatasetlabel=testdatasetlabel.to_numpy()
   for i in range(0,len(testdatasetlabel),timestep):
       listDte.append(testdatasetlabel[i])
       
   testdatasetlabel = np.array(listDte)
   
   
   
   
   
   list2 = []
   testlabel=testlabel.to_numpy()
   for i in range(0,len(testlabel),timestep):
       list2.append(testlabel[i])
       
   testy = np.array(list2)
   
   
   trainy = to_categorical(trainy)
   testy = to_categorical(testy) 
   
   return trainX,testX,trainy,testy,traindatasetlabel,testdatasetlabel



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



#%%#
def prepare_data_windows_without_split(trainX,trainlabel,traindatasetlabel,timestep):    
    
  
   remainTrain=len(trainX)%timestep

   
   if remainTrain>0:
       trainX= trainX[:-remainTrain]
       trainlabel= trainlabel[:-remainTrain]
       traindatasetlabel=traindatasetlabel[:-remainTrain]
   
   trainX= trainX.values.reshape((-1, timestep, trainX.shape[1]))

   #traindatasetlabel=traindatasetlabel.values.reshape((-1, timestep, trainX.shape[1]))
   #testdatasetlabel=testdatasetlabel.values.reshape((-1, timestep, testX.shape[1]))
   list1 = []
   trainlabel=trainlabel.to_numpy()
   for i in range(0,len(trainlabel),timestep):
       list1.append(trainlabel[i])
       
   trainy = np.array(list1)
   
   
   listDtr = []
   traindatasetlabel=traindatasetlabel.to_numpy()
   for i in range(0,len(traindatasetlabel),timestep):
       listDtr.append(traindatasetlabel[i])
       
   traindatasetlabel = np.array(listDtr)
   

   
   trainy = to_categorical(trainy)

   
   return trainX,trainy,traindatasetlabel



#%%#
# standardize data
def scale_data_without_split(trainX):
    # remove overlap
    cut = int(trainX.shape[1] / 2)
    longX = trainX[:, -cut:, :]
    # flatten windows
    longX = longX.reshape((longX.shape[0] * longX.shape[1], longX.shape[2]))
    # flatten train and test
    flatTrainX = trainX.reshape((trainX.shape[0] * trainX.shape[1], trainX.shape[2]))
   
    # standardize

    s = StandardScaler()
        # fit on training data
    s.fit(longX)
        # apply to training and test data
    longX = s.transform(longX)
    flatTrainX = s.transform(flatTrainX)
  
    # reshape
    flatTrainX = flatTrainX.reshape((trainX.shape))
 
    return flatTrainX


















