#!/usr/bin/env python
# script to create a pickle file with the dataset in a dictionary
# the script uses FRACTION_TRAINING for training and remaining as test data
# creates compatible dataset formats for multiple methods for comparison

import numpy as np
import pandas as pd
import pickle as pickle
import random
random.seed(123456789)          # fix for reproducibility

name = 'breast-cancer-wisconsin'
DATA_PATH = './'
FRACTION_TRAINING = 0.7

data = {}
filename = DATA_PATH + name + '.data'
df = pd.read_csv(filename,usecols=range(1,11),names=['thickness','size','shape','adhesion','se-size','nuclei','chromatin','nucleoli','mitoses','type'],na_values=['?'])
df.dropna(inplace=True)
x = df.drop(labels='type',axis=1).astype('int64').values
y = df['type'].astype('int64').values
y[y==2] = 0
y[y==4] = 1
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
# pickle.dump(data, open(DATA_PATH + name + ".pickle", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

# Make dataset with format compatible with SMC/MCMC method
data.pop('nx', None)
data.pop('ny', None)
data.pop('output_type',None)
data['n_class'] = len(np.unique(y)) 
data['n_dim'] = x.shape[1] 
data['is_sparse'] = False
# pickle.dump(data, open(DATA_PATH + name + ".p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

# # Make dataset with format compatible with Wu method
# # NOTE: needs to be scaled before running script -- divided by 10 here
# with open(DATA_PATH+name+'-train.txt', 'w') as file:
#     file.write('1\n') # 1 denotes classification
#     file.write(str(n_train)+'\n')
#     file.write(str(data['n_dim'])+'\n')
#     np.savetxt(file,np.hstack((data['x_train']/10,data['y_train'][:,None])),delimiter=' ',fmt='%.1f '*data['n_dim']+'%d')

# # NOTE: needs to be scaled before running script
# with open(DATA_PATH+name+'-test.txt', 'w') as file:
#     file.write('1\n') # 1 denotes classification
#     file.write(str(len(idx_test))+'\n')
#     file.write(str(data['n_dim'])+'\n')
#     np.savetxt(file,np.hstack((data['x_test']/10,data['y_test'][:,None])),delimiter=' ',fmt='%.1f '*data['n_dim']+'%d')

# ## Uncomment to create external CV datasets for other methods
# Create data splits for CV
from sklearn.model_selection import KFold
import copy
# Split training data into number of folds -- currently set to 5-fold
kf = KFold(n_splits=5,shuffle=True,random_state=1)

iii = 0
data_tmp = copy.deepcopy(data)
for train,test in kf.split(data['x_train']):
    # data_tmp['x_train'] = data['x_train'][train]
    # data_tmp['y_train'] = data['y_train'][train]
    # data_tmp['n_train'] = len(data_tmp['x_train'])
    # data_tmp['x_test'] = data['x_train'][test]
    # data_tmp['y_test'] = data['y_train'][test]
    # data_tmp['n_test'] = len(data_tmp['x_test'])
    # pickle.dump(data_tmp, open(DATA_PATH + name + "-CV-" + str(iii) + ".p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    with open(DATA_PATH+name+'-train-CV-'+str(iii)+'.txt', 'w') as file:
        file.write('1\n') # 1 denotes classification
        file.write(str(len(train))+'\n')
        file.write(str(data['n_dim'])+'\n')
        np.savetxt(file,np.hstack((data['x_train'][train]/10,data['y_train'][train,None])),delimiter=' ',fmt='%.1f '*data['n_dim']+'%d')
    with open(DATA_PATH+name+'-test-CV-'+str(iii)+'.txt', 'w') as file:
        file.write('1\n') # 1 denotes classification
        file.write(str(len(test))+'\n')
        file.write(str(data['n_dim'])+'\n')
        np.savetxt(file,np.hstack((data['x_train'][test]/10,data['y_train'][test,None])),delimiter=' ',fmt='%.1f '*data['n_dim']+'%d')
    iii = iii + 1 
