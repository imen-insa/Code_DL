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
vals = [funct_transfer(5,'ranao','nonconsec','all','uci') for _ in range(100)]

scores=stats_val(vals)

print(scores)

















