 #%%# 
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
import numpy as np 
import pandas as pd 
from random import randrange
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

def extract_sample(n_way, n_support, train_x, train_y):
  """
  random sample of size n_support+n_querry, for n_way classes
  Args:
      n_way (int): number of classes 
      n_support (int): number of examples per class in the support set
      train_x : dataset of activities
      train_y : dataset of labels
 
  """
  sample = []
  label=[]
  K = np.random.choice(np.unique(train_y), n_way, replace=False)
  for cls in K:
    datax_cls = train_x[train_y == cls]
    perm = np.random.permutation(datax_cls)
    sample_cls = perm[:(n_support)]
    label_cls=train_y[train_y == cls]
    label_cls = label_cls[:(n_support)]
    sample.append(sample_cls)
    label.append(label_cls)
  sample = np.array(sample)
  label = np.array(label)
  label=np.ravel(label)
  
  return sample,label


 #%%#  

def extract_sample_user(n_way, n_support, train_x, train_y,labelUser):
  """
  random sample of size n_support+n_querry, for n_way classes
  Args:
      n_way (int): number of classes 
      n_support (int): number of examples per class in the support set
      train_x : dataset of activities
      train_y : dataset of labels
 
  """
  sample = []
  label=[]
  
  W = np.random.choice(np.unique(labelUser), 1, replace=False)
  datax_cls1 = train_x[labelUser == W]
  
  
  K = np.random.choice(np.unique(train_y), n_way, replace=False)
  for cls in K:
    datax_cls = datax_cls1[train_y == cls]
    perm = np.random.permutation(datax_cls)
    sample_cls = perm[:(n_support)]
    label_cls=train_y[train_y == cls]
    label_cls = label_cls[:(n_support)]
    sample.append(sample_cls)
    label.append(label_cls)
  sample = np.array(sample)
  label = np.array(label)
  label=np.ravel(label)
  
  return sample,label

 
 #%%#  first consecutive samples

def extract_consec_sample(n_way, n_support, train_x, train_y):
  """
  random sample of size n_support+n_querry, for n_way classes
  Args:
      n_way (int): number of classes 
      n_support (int): number of examples per class in the support set
      train_x : dataset of activities
      train_y : dataset of labels
 
  """
  sample = []
  label=[]
  K = np.random.choice(np.unique(train_y), n_way, replace=False)
  for cls in K:
    datax_cls = train_x[train_y == cls]
    perm = datax_cls
    sample_cls = perm[:(n_support)]
    label_cls=train_y[train_y == cls]
    label_cls = label_cls[:(n_support)]
    sample.append(sample_cls)
    label.append(label_cls)
  sample = np.array(sample)
  label = np.array(label)
  label=np.ravel(label)
  
  return sample,label

#%%#
def extract_consecutive_sample(n_way, n_support, train_x, train_y):
  """
  random sample of size n_support+n_querry, for n_way classes
  Args:
      n_way (int): number of classes 
      n_support (int): number of examples per class in the support set
      train_x : dataset of activities
      train_y : dataset of labels
  Returns:
      numpy array and
      
      (dict) of:
        (torch.Tensor): sample of activities instances Size (n_way, n_support+n_query, 
                                                             (dim))
        (int): n_way
        (int): n_support
        (int): n_query
  """
  sample = []
  label=[]
  K = np.random.choice(np.unique(train_y), n_way, replace=False)
  for cls in K:
    datax_cls = train_x[train_y == cls]
    """"choose staring ramdom index"""
    long=len(datax_cls)-n_support
    randomOffset = randrange(0,long)
    
    sample_cls= datax_cls[randomOffset:randomOffset+n_support]
    
    label_cls=train_y[train_y == cls]
    label_cls = label_cls[:(n_support)]
    sample.append(sample_cls)
    label.append(label_cls)
  sample = np.array(sample)
  label = np.array(label)
  label=np.ravel(label)
  
  return sample,label 
