# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 09:12:05 2022

@author: trabelsi
"""
'read datasets'
#%%#
import os 
import numpy as np
import pandas as pd
from Other_Utililies import *
#from neurokit2.signal.signal_resample
from resampleSignal import * 

#%%#
def resample_data(df,sampling_rate,desired_sampling_rate):
    
    columns = list(df)
    metar_dat = dict()
    desired_length=None
    for i in columns:
     
        # resampling per column
        
        data_Column=nk.signal_resample(df.loc[:, i], 
                                       desired_length,sampling_rate,
                                       desired_sampling_rate)
        metar_dat[i] = data_Column
        
    new_data=pd.DataFrame(metar_dat)    
        

    
    return new_data
#%%#
def sample_dataset(data,actual_freq,desired_freq):
#create unique list of Activities
    UniqueNames = data.Activity.unique()
    UniqueCategories = data.Categories.unique()

#create a data frame dictionary to store your data frames
    DataFrameDict = {elem : pd.DataFrame for elem in UniqueNames}
    DataFrameDict1 = {elem : pd.DataFrame for elem in UniqueCategories}

    final_db = pd.DataFrame() #creates a new dataframe that's empty
    fusion=pd.DataFrame()
#or key, key1 in itertools.product(DataFrameDict.keys(),DataFrameDict1.keys()): 
    for key,key1 in zip(DataFrameDict.keys(),DataFrameDict1.keys()):   
        DataFrameDict[key] = data[:][data.Activity == key]
    
        New_data=DataFrameDict[key].iloc[:, -6:]
        New_data = New_data.sort_index()
   
        Data_prime=resample_data(New_data,actual_freq,desired_freq)
        df_label = pd.DataFrame(columns=['Activity'])
        array = np.empty(len(Data_prime), dtype = int)
        array.fill(key)
        array_label_pd = pd.DataFrame(data=array, columns=["Activity"])
    
    
    
        array_cat = np.empty(len(Data_prime), dtype = object)
        for i in range(len(Data_prime)):
            array_cat[i]=key1
    
        array_cat_pd = pd.DataFrame(data=array_cat, columns=["Categories"])
        fusion = pd.concat([array_label_pd,array_cat_pd,Data_prime], axis=1)
    
        final_db = final_db.append(fusion) #
        
    return final_db

#%%# UCIHAR 50 Hz
data_path ="./Final_data/UCI_Har_dataset.csv"
data=read_data(data_path)

result=sample_dataset(data,50,100)
result.to_csv('./data_resampled/UCI_dataset.csv', index = None)
#%%# PAMAP 100Hz: Nothing to do
  
    
#%%# Motion Sens dataset 50 Hz

data_path1 ="./Final_data/Motion_Sens_dataset.csv"
data1=read_data(data_path1)
result=sample_dataset(data1,50,100)

result.to_csv('./data_resampled/Motion_Sens_dataset.csv', index = None)    
    
#%%# w-Har dataset 250 Hz

data_path2 ="./Final_data/wHar_dataset.csv"
data2=read_data(data_path2)


result=sample_dataset(data2,250,100)
                
result.to_csv('./data_resampled/Motion_Sens_dataset.csv', index = None)                       
   
    
