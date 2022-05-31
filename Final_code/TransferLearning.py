# -*- coding: utf-8 -*-
"""
transfert without few shots on UCI har

@author: trabelsi


    #%%#
"""
#%%#
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Model, model_from_json

import os
from extract_few_sample import * 
from DataActivity import *


 #%%# load data from UCI
CUR_DIR = os.getcwd() # Path to current directory

LOG_DIR = os.path.join(PAR_DIR, "log")

PAR_DIR= os.path.dirname(CUR_DIR)

DATA_DIR = os.path.join(PAR_DIR, "data_resampled")


data_path1 = os.path.join(DATA_DIR, "UCI_Har_dataset.csv")
#%%# load models
# load json and create model
def load_model_Ranao():
    json_file = open('modelRanao_json.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
# load weights into new model
    loaded_model.load_weights("my_modelRanao_weights.h5")
    return loaded_model


#%%# 
def load_model_Ignatov():
    json_file = open('modelIgnatov_json.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
# load weights into new model
    loaded_model.load_weights("my_modelIgnatov_weights.h5")
    return loaded_model

#%%# 
def load_model_Ordonez():
    json_file = open('modelOrdonez_json.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
# load weights into new model
    loaded_model.load_weights("my_modelOrdonez_weights.h5")
    return loaded_model

  #%%# 
def load_data(dataset_name):
    if dataset_name=="uci":
        DATA_DIR = os.path.join(PAR_DIR, "data_resampled")
    
        data_path1 = os.path.join(DATA_DIR, "UCI_Har_dataset.csv")
    
        train,test,trainlabel,testlabel,traindatasetlabel,testdatasetlabel=prepare_data(data_path1)
        train_UCI,test_UCI,trainy_UCI,testy_UCI=prepare_data_windows(train,test,trainlabel,testlabel,50)
        train_UCI,test_UCI=scale_data(train_UCI,test_UCI)
        train_UCI = train_UCI.reshape((train_UCI.shape[0], train_UCI.shape[1], train_UCI.shape[2],1))
        test_UCI = test_UCI.reshape((test_UCI.shape[0], test_UCI.shape[1], test_UCI.shape[2],1))
    else:
        raise Exception(f"Only on UCI")
    return train_UCI, test_UCI,trainy_UCI,testy_UCI

  #%%# 

def load_data_user(dataset_name):
    if dataset_name=="uci":
        DATA_DIR = os.path.join(PAR_DIR, "data_resampled")
    
        data_path1 = os.path.join(DATA_DIR, "UCI_Har_dataset.csv")
    
     
        train,test,trainlabel,testlabel,trainUser,testUser=prepare_data_user(data_path1)
        
        train_UCI,test_UCI,trainy_UCI,testy_UCI=prepare_data_windows(train,test,trainlabel,testlabel,50)
        train_UCI,test_UCI=scale_data(train_UCI,test_UCI)

        train_UCI = train_UCI.reshape((train_UCI.shape[0], train_UCI.shape[1], train_UCI.shape[2],1))
        test_UCI = test_UCI.reshape((test_UCI.shape[0], test_UCI.shape[1], test_UCI.shape[2],1))
    else:
        raise Exception(f"Only on UCI")
    return train_UCI, test_UCI,trainy_UCI,testy_UCI




 #%%# 
def meanstd(listvalue):
    val_mean=np.mean(listvalue)

    val_std=np.std(listvalue)
    return val_mean,val_std
def stats_val(vals):
    val_acc = [data_dict['accuracy'] for data_dict in vals]
    val_log = [data_dict['loss'] for data_dict in vals]
    val_prec = [data_dict['precision'] for data_dict in vals]
    val_recall = [data_dict['recall'] for data_dict in vals]
    val_f1 = [data_dict['f1'] for data_dict in vals]
    
    acc_mean,acc_std=meanstd(val_acc)
    
    log_mean,log_std=meanstd(val_log)
    
    prec_mean,prec_std=meanstd(val_prec)
    
    recall_mean,recall_std=meanstd(val_recall)
    
    f1_mean,f1_std=meanstd(val_f1)
    
    
    scores: Dict[str, Dict[str, List[Any]]] = {
        "accuracy_mean": [],
        "accuracy_std": [],
        "loss_mean": [],
        "loss_std": [],
        "precision_mean": [],
        "precision_std": [],
        "recall_mean": [],
        "recall_std": [],
        "f1_mean": [],
        "f1_std": [],
        "recall_mean": [],
    }
    
    scores["accuracy_mean"].append(acc_mean)
    scores["accuracy_std"].append(acc_std)
    scores["loss_mean"].append(log_mean)
    scores["loss_std"].append(log_std)
    scores["precision_mean"].append(prec_mean)
    scores["precision_std"].append(prec_std)
    scores["recall_mean"].append(recall_mean)
    scores["recall_std"].append(recall_std)
    scores["f1_mean"].append(f1_mean)
    scores["f1_std"].append(f1_std)
    return scores



#%%# 
def funct_transfer(nshot,base_model,mode_frame, mode_user, dataset_name):
    
    if mode_user=='one':
        train_UCI, test_UCI,trainy_UCI,testy_UCI=load_data_user(dataset_name)
        labelTrain=np.argmax(trainy_UCI,axis=1)
        
    
    elif mode_user=='all': 
 
         train_UCI,test_UCI,trainy_UCI,testy_UCI=load_data(dataset_name)
         labelTrain=np.argmax(trainy_UCI,axis=1)
         
    else:   
         raise Exception(f"Incoherent mode")
        
        
        
    if mode_frame=='consec':

        sample,label = extract_consecutive_sample(5,nshot,train_UCI,labelTrain)
    elif mode_frame=='nonconsec':
        sample,label = extract_sample(5,nshot,train_UCI,labelTrain)
    else:
        raise Exception(f"Incoherent mode")
    sample = sample.reshape((sample.shape[0] * sample.shape[1], sample.shape[2],sample.shape[3]))
        
    label = to_categorical(label)
        
    sample = sample.reshape((sample.shape[0], sample.shape[1], sample.shape[2],1))
        
    """ Freeze the convolutional base:
     Freezing (by setting layer.trainable = False) prevents the weights
     in a given layer from being updated during training.
     #%%# test on test data from Ranao model with transfer ( train 70, test 30)
    """ 
    drp_out_dns = 0.5
    nb_dense = 128
    batch_size=64
        
        
        
    inputs = keras.Input(shape=(50,6,1))
        # base_model.include_top=False # Do not include classifier at the top
        
    if base_model=="ranao":
        base_model=load_model_Ranao()
          
    elif base_model=="ordonez":
        base_model=load_model_Ordonez()
           
            
    elif base_model=="ignatov":
         base_model=load_model_Ignatov()
             
    else:
        raise Exception(f"Incoherent model")   
        
        
        
        
        
    base_model.trainable = True #  freeze the model
    for l in base_model.layers[:-3]:
        l.trainable = False
            
    score=train_and_predict(base_model,sample,label,test_UCI,testy_UCI,epochs=50,verbose=1, batch_size=64)
    return score
#%%# 
def funct_witout_transfer(nshot,base_model,mode_frame,mode_user,dataset_name):
    n_classes=5
    if mode_user=='one':
        train_UCI, test_UCI,trainy_UCI,testy_UCI=load_data_user(dataset_name)
        labelTrain=np.argmax(trainy_UCI,axis=1)
        
    
    elif mode_user=='all': 
 
         train_UCI,test_UCI,trainy_UCI,testy_UCI=load_data(dataset_name)
         labelTrain=np.argmax(trainy_UCI,axis=1)
         
    else:   
         raise Exception(f"Incoherent mode")
        
        
        
    if mode_frame=='consec':

        sample,label = extract_consecutive_sample(5,nshot,train_UCI,labelTrain)
    elif mode_frame=='nonconsec':
        sample,label = extract_sample(5,nshot,train_UCI,labelTrain)
    else:
        raise Exception(f"Incoherent mode")
    sample = sample.reshape((sample.shape[0] * sample.shape[1], sample.shape[2],sample.shape[3]))
        
    label = to_categorical(label)
        
    sample = sample.reshape((sample.shape[0], sample.shape[1], sample.shape[2],1))
        
    
    drp_out_dns = 0.5
    nb_dense = 128
    batch_size=64
        
        
        
    inputs = keras.Input(shape=(50,6,1))
        # base_model.include_top=False # Do not include classifier at the top
        
    if base_model=="ranao":
        base_model=build_model_ronao(sample, label, n_classes)
        score=train_and_predict(base_model,sample,label,test_UCI,testy_UCI,epochs=50,verbose=1, batch_size=64)
        

          
    elif base_model=="ordonez":
        base_model=build_model_ordonez(sample, label, n_classes)
        score=train_and_predict(base_model,sample,label,test_UCI,testy_UCI,epochs=50,verbose=1, batch_size=64)
        
      
            
    elif base_model=="ignatov":
        base_model=build_model_ignatov(sample, label,n_classes)
        score=train_and_predict(base_model,sample,label,test_UCI,testy_UCI,epochs=50,verbose=1, batch_size=64)
        
         
             
    else:
        raise Exception(f"Incoherent model")   
          
    return score

 

                                 