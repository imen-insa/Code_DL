# -*- coding: utf-8  -*-
# -*- Uniform data and labels for all datasets

import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder
#%%# UCIHAR 50 Hz


data_path ="./data_resampled/UCI_Har_dataset.csv"

data=read_data(data_path)

UniqueCategories = data.Categories.unique()
print(UniqueCategories)

UCI_Har= data[data['Categories'].isin(['WALKING', 'SITTING','STANDING','WALKING_DOWNSTAIRS',
 'WALKING_UPSTAIRS'])]


UniqueCategories = UCI_Har.Categories.unique()
print(UniqueCategories)




UCI_Har['Categories'] = UCI_Har['Categories'].replace(['STANDING','SITTING', 'WALKING', 'WALKING_DOWNSTAIRS',
                                                       'WALKING_UPSTAIRS'],
                                                    ['standing','sitting','walking','descending',
                                                     'ascending'])
UCI_Har['Label_act']=LabelEncoder().fit_transform(UCI_Har['Categories'])

UCI_Har=UCI_Har[['User', 'Activity', 'Categories', 'Label_act','Acc-x','Acc-y','Acc-z','Gyr-x', 'Gyr-y','Gyr-z']]

UCI_Har.to_csv('./data_resampled/UCI_Har_dataset.csv', index = None)

#%%# MotionSens 50 Hz        
data_path ="./data_resampled/Motion_Sens_dataset.csv"

data=read_data(data_path)

UniqueCategories = data.Categories.unique()
print(UniqueCategories)
['DOWNSTAIRS' 'UPSTAIRS' 'WALKING' 'JOGING' 'STANDING' 'SITTING']
UCI_Sens=data[data.Categories != 'JOGING']


UniqueCategories = UCI_Sens.Categories.unique()

print(UniqueCategories)


UCI_Sens['Categories'] = UCI_Sens['Categories'].replace(['DOWNSTAIRS', 'UPSTAIRS', 'WALKING',
                                                         'STANDING', 'SITTING'],
                                                        ['descending', 'ascending','walking',
                                                         'standing','sitting'])
UCI_Sens['Label_act']=LabelEncoder().fit_transform(UCI_Sens['Categories'])

UCI_Sens=UCI_Sens[['User', 'Activity', 'Categories', 'Label_act','Acc-x','Acc-y','Acc-z','Gyr-x', 'Gyr-y','Gyr-z']]
UCI_Sens.to_csv('./data_resampled/Motion_Sens_dataset.csv', index = None)

#%%# wHar dataset 250 Hz        
       
data_path ="./data_resampled/wHar_dataset.csv"

data=read_data(data_path)

UniqueCategories = data.Categories.unique()
print(UniqueCategories)


['walk' 'transition' 'sit' 'stand' 'jump' 'undefined' 'liedown' 'stairsup'
 'stairsdown']


UCI_wHar= data[data['Categories'].isin(['walk', 'sit','stand','stairsup',
                                        'stairsdown']) ]

UCI_wHar['Categories'] = UCI_wHar['Categories'].replace(['walk', 'sit','stand','stairsup','stairsdown'],
                                                    ['walking','sitting','standing',
                                                     'ascending','descending'])



UniqueCategories = UCI_wHar.Categories.unique()
print(UniqueCategories)


UCI_wHar['Label_act']=LabelEncoder().fit_transform(UCI_wHar['Categories'])

UCI_wHar=UCI_wHar[['User', 'Activity', 'Categories', 'Label_act','Acc-x','Acc-y','Acc-z','Gyr-x', 'Gyr-y','Gyr-z']]
UCI_wHar.to_csv('./data_resampled/wHar_dataset.csv', index = None)
#%%# PAMAP dataset 100 Hz   

data_path ="./data_resampled/Pamap_dataset.csv"

data=read_data(data_path)

UniqueCategories = data.Categories.unique()
print(UniqueCategories)
['other' 'lying' 'descending stairs' 'ascending stairs' 'cycling'
 'Nordic walking' 'sitting' 'standing' 'rope jumping']



UCI_Pamap= data[data['Categories'].isin(['Nordic walking', 'sitting', 'standing', 
                                       'descending stairs' ,'ascending stairs']) ]


UniqueCategories = UCI_Pamap.Categories.unique()
print(UniqueCategories)

UCI_Pamap['Categories'] = UCI_Pamap['Categories'].replace(['Nordic walking', 'sitting', 'standing', 
                                       'descending stairs' ,'ascending stairs'],
                                                    ['walking','sitting','standing',
                                                     'descending','ascending'])

UCI_Pamap['Label_act']=LabelEncoder().fit_transform(UCI_Pamap['Categories'])

UCI_Pamap=UCI_Pamap[['User', 'Activity', 'Categories', 'Label_act','Acc-x','Acc-y','Acc-z','Gyr-x', 'Gyr-y','Gyr-z']]
UCI_Pamap.to_csv('./data_resampled/Pamap_dataset.csv', index = None)

#%%# Wisdm dataset 20 Hz   

data_path ="./data_resampled/Wisdm_dataset_phone.csv"
data=read_data(data_path)

UniqueCategories = data.Categories.unique()
print(UniqueCategories)


[' walking' 'jogging' 'stairs' 'sitting' 'standing' 'typing' 'teeth'
 'soup' 'chips' 'pasta' 'drinking' 'sandwich ' 'kicking' 'catch '
 'dribbling ' 'writing' 'clapping' 'folding']



UCI_Wisdm= data[data['Categories'].isin(['walking', 'sitting', 'standing',]) ]

UCI_Wisdm['Label_act']=LabelEncoder().fit_transform(UCI_Wisdm['Categories'])

UCI_Wisdm=UCI_Wisdm[['User', 'Activity', 'Categories', 'Label_act','Acc-x','Acc-y','Acc-z','Gyr-x', 'Gyr-y','Gyr-z']]
UCI_Wisdm.to_csv('./data_resampled/Wisdm_dataset_phone.csv', index = None)

#%%# Wisdm dataset 20 Hz   watch

data_path ="./data_resampled/Wisdm_dataset_watch.csv"
data=read_data(data_path)

UniqueCategories = data.Categories.unique()
print(UniqueCategories)


[' walking' 'jogging' 'stairs' 'sitting' 'standing' 'typing' 'teeth'
 'soup' 'chips' 'pasta' 'drinking' 'sandwich ' 'kicking' 'catch '
 'dribbling ' 'writing' 'clapping' 'folding']



UCI_Wisdm= data[data['Categories'].isin(['walking', 'sitting', 'standing',]) ]

UCI_Wisdm['Label_act']=LabelEncoder().fit_transform(UCI_Wisdm['Categories'])

UCI_Wisdm=UCI_Wisdm[['User', 'Activity', 'Categories', 'Label_act','Acc-x','Acc-y','Acc-z','Gyr-x', 'Gyr-y','Gyr-z']]
UCI_Wisdm.to_csv('./data_resampled/Wisdm_dataset_watch.csv', index = None)
