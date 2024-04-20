#!/usr/bin/env python
# script to create a pickle file with the dataset in a dictionary
# the script uses FRACTION_TRAINING for training and remaining as test data

import numpy as np
import pandas as pd
import pickle as pickle
import random
random.seed(123456789)          # fix for reproducibility

name = 'wine'
DATA_PATH = './'
FRACTION_TRAINING = 0.8

data = {}
filename = DATA_PATH + name + '.data'
df = pd.read_csv(filename,names=['wine','alc','mal-acid','ash','alc-ash','mag','tot-ph','flav','nonflav','proanth','color','hue','od','proline'],index_col=False)
df['wine']=df['wine'].astype('category').cat.codes
x = df.drop(labels='wine',axis=1).values
y = df['wine'].values
n = x.shape[0]
print('ndim = ',x.shape[1])

data['nx'] = x.shape[1] 
idx = [i for i in range(n)]
random.shuffle(idx)

n_train = int(FRACTION_TRAINING * n)
idx_train = idx[:n_train]
idx_test = idx[n_train:]
data['n_train'] = n_train
data['n_test'] = len(idx_test)
data['y_train'] = y[idx_train]
data['y_test'] = y[idx_test]
data['x_train'] = x[idx_train,:]
data['x_test'] = x[idx_test,:]
data['ny'] = len(np.unique(y)) 
data['output_type'] = 'class'

# Dataset format required for HMC
pickle.dump(data, open(DATA_PATH + name + ".pickle", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

# Make dataset with format compatible with SMC/MCMC method
data.pop('nx', None)
data.pop('ny', None)
data.pop('output_type',None)
data['n_class'] = len(np.unique(y)) 
data['n_dim'] = x.shape[1] 
data['is_sparse'] = False
pickle.dump(data, open(DATA_PATH + name + ".p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

# NOTE: Wu method only compatible with classification datasets with two outputs -- incompatible with wine data

# Create data splits for CV
from sklearn.model_selection import KFold
import copy
# Split training data into number of folds -- currently set to 5-fold
kf = KFold(n_splits=5,shuffle=True,random_state=1)

iii = 0
data_tmp = copy.deepcopy(data)
for train,test in kf.split(data['x_train']):
    data_tmp['x_train'] = data['x_train'][train]
    data_tmp['y_train'] = data['y_train'][train]
    data_tmp['n_train'] = len(data_tmp['x_train'])
    data_tmp['x_test'] = data['x_train'][test]
    data_tmp['y_test'] = data['y_train'][test]
    data_tmp['n_test'] = len(data_tmp['x_test'])
    pickle.dump(data_tmp, open(DATA_PATH + name + "-CV-" + str(iii) + ".p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    iii = iii + 1  