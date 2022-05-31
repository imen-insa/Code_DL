# -*- coding: utf-8 -*-
"""
Created on Mon May 23 14:05:41 2022

@author: trabelsi
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May  9 19:09:11 2022

@author: trabelsi
"""

"""Functions for training Convolutional Neural Network (CNN)"""
#%%#
from typing import Any, Dict, List
import numpy as np
import os
import pandas as pd
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from src.utils import plot_learning_history, plot_model
from keras_callback import create_callback
from base1 import *
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
CUR_DIR = os.getcwd() # Path to current directory
PAR_DIR= os.path.dirname(CUR_DIR)
LOG_DIR = os.path.join(PAR_DIR, "log")
#%%#


def train_and_predict(model, ds_train, train_y, ds_test, test_y, epochs, verbose,batch_size):
    """Train CNN
    Args:
        X_train, X_valid, X_test: input signals of shape
            (num_samples, window_size, num_channels, 1)
        y_train, y_valid, y_test: onehot-encoded labels
    Returns:
        pred_train: train prediction
        pred_valid: train prediction
        pred_test: train prediction
        model: trained best model
    """

    """input_shape, output_shape = [(x.shape, y.shape) for x, y in ds_train.take(1)][0]"""
    
    input_shape= keras.Input(shape=(50,6,1))

    output_shape = ds_train.shape[1]

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    with open(f"{LOG_DIR}/model_summary.txt", "w") as fh:
        model.summary(print_fn=lambda x: fh.write(x + "\n"))
    plot_model(model, path=f"{LOG_DIR}/model.png")

    callbacks = create_callback(log_dir=LOG_DIR,verbose=verbose,epochs=epochs)

  
    
    fit=model.fit(ds_train, train_y, epochs=epochs,
                           batch_size=batch_size,callbacks=callbacks)
    
    
    

    """plot_learning_history1(fit=fit, path=f"{LOG_DIR}/history.png")"""
    """best_model = keras.models.load_model(f"{LOG_DIR}/trained_model.h5")"""
    best_model=model

    pred_train = model.predict(ds_train)
    pred_test = model.predict(ds_test)

    scores: Dict[str, Dict[str, List[Any]]] = {
        "loss": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "cm": [],
        "per_class_f1": [],
    }
 
    
    for pred, ds in zip(
        [ pred_test],
        [ds_test],
    
    ):
        
        
        loss, acc= best_model.evaluate(ds_test, test_y, verbose=0)
        
        pred_test1 = pred_test.argmax(axis=1)
       
        y=test_y
        y = y.argmax(axis=1)
        
        
        
        
        "y = np.argmax(y,axis=1)"
        scores["loss"].append(loss)
        scores["accuracy"].append(acc)
        scores["precision"].append(precision_score(y, pred_test1, average="macro"))
        scores["recall"].append(recall_score(y, pred_test1, average="macro"))
        scores["f1"].append(f1_score(y, pred_test1, average="macro"))
        scores["cm"].append(confusion_matrix(y, pred_test1, normalize="true"))
        scores["per_class_f1"].append(f1_score(y, pred_test1, average=None))

    np.save(f"{LOG_DIR}/scores.npy", scores)
    df_scores = pd.DataFrame([])
   
    res_dict = {}
    for metric in ["loss", "accuracy", "precision", "recall", "f1"]:
        res_dict[metric] = [round(np.mean(scores[metric]))]
    df = pd.DataFrame(res_dict)
    df_scores = pd.concat([df_scores, df])
    print("--------------------------------------------------------")
    print(df_scores)
    print("--------------------------------------------------------")
    df_scores.to_csv(f"{LOG_DIR}/scores.csv")
    np.save(f"{LOG_DIR}/test_oof.npy", np.mean(pred_test, axis=0))

    # Plot confusion matrix
   

    return scores









