import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import matplotlib.pyplot as plt

import datetime

import torch
import torch.nn as nn
from torch.autograd import Variable 

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# import datset
df = pd.read_csv('./Pytorch/PV_Elec_Gas3.csv').rename(columns={'date':'timestamp'}).set_index('timestamp')
# for param in df.columns:
#     df[param] = df[param].astype('float32').dtypes

print('*'*50)
print('data : \n{}'.format(df)) 
print('*'*50)
print('\ndata type : \n{}'.format(df.dtypes))
print('*'*50)

# split dataset to train/valid
train_set = df[:'31/10/2018']
valid_set = df['1/11/2018':'18/11/2019']

# show the proportion of each dataset
print('Proportion of train_set : {:.2f}%'.format(len(train_set)/len(df)))
print('Proportion of valid_set : {:.2f}%'.format(len(valid_set)/len(df)))

# The split_sequence() function below will split a given univariate sequence into multiple samples 
# where each sample has a specified number of time steps and the output is a single time step.
def split_sequence(sequence, n_steps):
    x, y = list(), list()
    for i in range(len(sequence)):
        
        end_ix = i + n_steps
        
        if end_ix > len(sequence)-1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        x.append(seq_x)
        y.append(seq_y)
    return array(x), array(y)

n_steps = 3
train_x,train_y = split_sequence(train_set['Gas/day'].values,n_steps)
valid_x,valid_y = split_sequence(valid_set['Gas/day'].values,n_steps)

# summarize the data
print('*'*50)
print("train_x, train_y splitted  : ")
for i in range(10):
	print(train_x[i], train_y[i])
print('*'*50)