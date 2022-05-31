# -*- coding: utf-8 -*-
"""
Created on Mon May  9 11:16:48 2022

@author: trabelsi
"""
 #%%#
import numpy as np  
import json
from keras.models import Model, model_from_json
from logging import basicConfig, getLogger, StreamHandler, DEBUG, WARNING
import os
import sys
import argparse
from tensorflow.keras import regularizers
from shutil import copyfile
import numpy as np
import pandas as pd
from inspect import getsource
from src.load_dataset import load_dataset  
from TransferLearning import *
from extract_few_sample import * 
from DataActivity import *


LOG_DIR = os.path.join(PAR_DIR, "log")

#%%# N shots on uci har ( consecutive or nonconsecutive frames)

"vals = [funct_transfer(number_shots,model,mode_frame,mode_user, dataset) for _ in range(iteraton_number)]"


"model: ranao, ignatov,ordonez"
"mode_frame:consec or noncosec"
"mode_user:one or all"
"dataset: uci"


vals = [funct_transfer(5,'ranao','nonconsec','all','uci') for _ in range(2)]

scores=stats_val(vals)

print(scores)


























#%%# 5 fives shots user specific recognition
labelTrain=np.argmax(trainy_UCI,axis=1)

sample,label = extract_sample(5, 500,train_UCI,labelTrain)


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

base_model=load_model_final()

inputs = keras.Input(shape=(50,6,1))
# base_model.include_top=False # Do not include classifier at the top
base_model.trainable = True #  freeze the model
for l in base_model.layers[:-3]:
    l.trainable = False






#callbacks = create_callback(
 #   log_dir=LOG_DIR,
  #  verbose=1,
 #   epochs=50,
   
#)


scores=train_and_predict(base_model,sample,label,test_UCI,testy_UCI,epochs=50,verbose=1, batch_size=64)




#print("%s: %.2f%%" % (base_model.metrics_names[1], score[1]*100))

""" accuracy : 85.53%"""

"""plot_learning_history(fit=history path=f"{LOG_DIR}/history.png")"""

#%%# consecutive shots



labelTrain=np.argmax(trainy_UCI,axis=1)

sample,label = extract_consec_sample(5,1,train_UCI,labelTrain)


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

base_model=load_model_final()

inputs = keras.Input(shape=(50,6,1))
# base_model.include_top=False # Do not include classifier at the top
base_model.trainable = True #  freeze the model
for l in base_model.layers[:-3]:
    l.trainable = False






#callbacks = create_callback(
 #   log_dir=LOG_DIR,
  #  verbose=1,
 #   epochs=50,
   
#)


scores=train_and_predict(base_model,sample,label,test_UCI,testy_UCI,epochs=50,verbose=1, batch_size=64)




#print("%s: %.2f%%" % (base_model.metrics_names[1], score[1]*100))

""" accuracy : 85.53%"""

"""plot_learning_history(fit=history path=f"{LOG_DIR}/history.png")"""

























#%%#fine tunning
""" we start by unfreezing all layers of the base model"""
new_model=load_model_final()
base_model=new_model
base_model.trainable = True
base_model.include_top=False 
# Freeze all layers except the fine_tune_at layer
fine_tune_at=2
for layer in base_model.layers[:-fine_tune_at]: 
    layer.trainable = False

# compile and retrain with a low learning rate




inputs = keras.Input(shape=(50,6,1))


 # very low learning rate at this stage, we are training a much larger model on a very small dataset 
opt = keras.optimizers.Adam(learning_rate=0.000001)

kernel_regularizer = regularizers.l2(0.00005)


base_model.add(
        Dense(
            nb_dense,
            kernel_regularizer=kernel_regularizer,
            activation="relu",
          
        )
    )
base_model.add(Dropout(drp_out_dns))
base_model.add(Dense(5, activation="softmax"))
# base_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# We make sure that the base_model is running in inference mode here,
# by passing `training=False`. T
 # fit network
# history=base_model.fit(train_UCI, trainy_UCI, epochs=30, batch_size=batch_size, verbose=0)

 # evaluate model
 
# score = base_model.evaluate(test_UCI, testy_UCI, batch_size=batch_size, verbose=0)






scores=train_and_predict(base_model,sample,label,test_UCI,testy_UCI,epochs=50,verbose=0, batch_size=64)


print("%s: %.2f%%" % (base_model.metrics_names[1], score[1]*100))

#%%#fine tunning



base_model=loaded_model
base_model.trainable = True
base_model.include_top=False 
inputs = keras.Input(shape=(50,6,1))
output = loaded_model.layers[-1].output
output = keras.layers.Flatten()(output)
base_model1 = Model(inputs, output)






