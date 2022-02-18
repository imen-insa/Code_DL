# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 09:12:05 2022

@author: trabelsi
"""
'resampling all datasets'
#%%#
import os 
import numpy as np
import pandas as pd
from Other_Utililies import *
from resampleSignal import * 
import neurokit2 as nk # install neurokit2:  pip install neurokit2 
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

def duplicate_labels(df, n_times):
    ndf = pd.DataFrame(np.repeat(df.values, n_times,axis=0) )
    ndf.columns = df.columns
    return ndf



#%%#
def resample_labels(df, sampling_rate,desired_n_times):
    


     n_times=desired_n_times/sampling_rate
     result=pd.DataFrame()
    
    
     if n_times>=1:
    
      for i in range(0, len(df), sampling_rate):

         A= df[:][i:i+sampling_rate]
         
         B= pd.DataFrame(np.repeat(A,n_times) )
         result=pd.concat([result,B])
         
               
    
     else:    
      for i in range(0, len(df), sampling_rate):

         A= df[:][i:i+sampling_rate]
         B=A.head(desired_n_times)
         result=pd.concat([result,B])    
         
     return result
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

    for key,key1 in zip(DataFrameDict.keys(),DataFrameDict1.keys()):   
        DataFrameDict[key] = data[:][data.Activity == key]
        DataFrameDict[key] = DataFrameDict[key].sort_index()
    
        New_data=DataFrameDict[key].iloc[:, -6:]
        User_data=DataFrameDict[key].loc[:,'User']
        #User_data=DataFrameDict[key].iloc[:,0]
        #User_data = User_data.sort_index()
        User_data=resample_labels(User_data,actual_freq,desired_freq)
        User_data=User_data.reset_index(drop=True)
        Data_prime=resample_data(New_data,actual_freq,desired_freq)
       
        
        df_label = pd.DataFrame(columns=['Activity'])
        array = np.empty(len(Data_prime), dtype = int)
        array.fill(key)
        array_label_pd = pd.DataFrame(data=array, columns=["Activity"])
    
    
    
        array_cat = np.empty(len(Data_prime), dtype = object)
        for i in range(len(Data_prime)):
            array_cat[i]=key1
    
        array_cat_pd = pd.DataFrame(data=array_cat, columns=["Categories"])
        fusion = pd.concat([User_data,array_label_pd,array_cat_pd,Data_prime], axis=1)
    
        final_db = final_db.append(fusion) #
        
    return final_db

#%%# UCIHAR 50 Hz


data_path ="./Final_data/UCI_Har_dataset.csv"
data=read_data(data_path)
result=sample_dataset(data,50,100)
result.to_csv('./data_resampled_all/UCI_Har_dataset.csv', index = None)
#%%# PAMAP 100Hz: Nothing to do
  
    
#%%# Motion Sens dataset 50 Hz

data_path1 ="./Final_data/Motion_Sens_dataset.csv"
data1=read_data(data_path1)
result=sample_dataset(data1,50,100)

result.to_csv('./data_resampled_all/Motion_Sens_dataset.csv', index = None)    
    
#%%# w-Har dataset 250 Hz

data_path2 ="./Final_data/wHar_dataset.csv"
data2=read_data(data_path2)


result=sample_dataset(data2,250,100)
                
result.to_csv('./data_resampled_all/wHar_dataset.csv', index = None)                       
   
    
# wisdm watch dataset 20 Hz

data_path2 ="./Final_data/Wisdm_dataset_watch.csv"
data2=read_data(data_path2)
result=sample_dataset(data2,20,100)
result.to_csv('./data_resampled_all/Wisdm_dataset_watch.csv', 
              index = None) 


# wisdm phone dataset 20 Hz
data_path2 ="./Final_data/Wisdm_dataset_phone.csv"
data2=read_data(data_path2)


result=sample_dataset(data2,20,100)
                
result.to_csv('./data_resampled_all/Wisdm_dataset_phone.csv', index = None) 















 
