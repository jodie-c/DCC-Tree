# script used to create wu dataset
# creates mulitple versions compatible with different methods

import numpy as np
import pandas as pd
import pickle as pickle
import random

random.seed(1)

dataset = 'wu'

def generate_wu_dataset(): # creates in HMC compatible format
    """ Generates the synthetic data as described in Section 5.1 of (Wu, et al. 2007). """
    p = 3 # number of predictors
    n_train = 300 # number of training observations
    n_test = 300 # number of testing observations

    x_train = np.zeros([n_train,p])
    y_train = np.zeros(n_train)
    x_test = np.zeros([n_test,p])
    y_test = np.zeros(n_test)

    # create training part of dataset
    x_train[:200,0] = np.random.uniform(0.1,0.4,200)
    x_train[:200,2] = np.random.uniform(0.6,0.9,200)
    x_train[:100,1] = np.random.uniform(0.1,0.4,100)
    x_train[100:200,1] = np.random.uniform(0.6,0.9,100)
    x_train[200:,0] = np.random.uniform(0.6,0.9,100)
    x_train[200:,1] = np.random.uniform(0.1,0.9,100)
    x_train[200:,2] = np.random.uniform(0.1,0.4,100)   
            
    y_train[:100] = 1 + np.random.normal(0,0.25,100)
    y_train[100:200] = 3 + np.random.normal(0,0.25,100)
    y_train[200:] = 5 + np.random.normal(0,0.25,100)

    # create testing part of dataset
    x_test[:200,0] = np.random.uniform(0.1,0.4,200)
    x_test[:200,2] = np.random.uniform(0.6,0.9,200)
    x_test[:100,1] = np.random.uniform(0.1,0.4,100)
    x_test[100:200,1] = np.random.uniform(0.6,0.9,100)
    x_test[200:,0] = np.random.uniform(0.6,0.9,100)
    x_test[200:,1] = np.random.uniform(0.1,0.9,100)
    x_test[200:,2] = np.random.uniform(0.1,0.4,100)   
            
    y_test[:100] = 1 + np.random.normal(0,0.25,100)
    y_test[100:200] = 3 + np.random.normal(0,0.25,100)
    y_test[200:] = 5 + np.random.normal(0,0.25,100)
    
    data = {'x_train': x_train, 'y_train': y_train, \
            'nx': p, 'ny': 1, 'n_train': n_train, 'x_test': x_test, \
            'y_test': y_test, 'n_test': n_test, 'output_type':'real'}
    return data

data = generate_wu_dataset()

# Dataset format required for HMC
pickle.dump(data, open(dataset + ".pickle", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

# NOTE: dataset not compatible with SMC method

# Make dataset with format compatible with Wu/CGM methods
with open(dataset+'-train.txt', 'w') as file:
    file.write('0\n') # 0 denotes regression
    file.write(str(data['n_train'])+'\n')
    file.write(str(data['nx'])+'\n')
    np.savetxt(file,np.hstack((data['x_train'],data['y_train'][:,None])),delimiter=' ',fmt='%.3f '*data['nx']+'%.3f')

with open(dataset+'-test.txt', 'w') as file:
    file.write('0\n') # 0 denotes regression
    file.write(str(data['n_test'])+'\n')
    file.write(str(data['nx'])+'\n')
    np.savetxt(file,np.hstack((data['x_test'],data['y_test'][:,None])),delimiter=' ',fmt='%.3f '*data['nx']+'%.3f')

# TODO UNTESTED CODE BELOW
# # ## Uncomment to create external CV datasets for other methods
# # Create data splits for CV
# from sklearn.model_selection import KFold
# import copy
# # Split training data into number of folds -- currently set to 5-fold
# kf = KFold(n_splits=5,shuffle=True,random_state=1)

# iii = 0
# data_tmp = copy.deepcopy(data)
# for train,test in kf.split(data['x_train']):
#     # data_tmp['x_train'] = data['x_train'][train]
#     # data_tmp['y_train'] = data['y_train'][train]
#     # data_tmp['n_train'] = len(data_tmp['x_train'])
#     # data_tmp['x_test'] = data['x_train'][test]
#     # data_tmp['y_test'] = data['y_train'][test]
#     # data_tmp['n_test'] = len(data_tmp['x_test'])
#     # pickle.dump(data_tmp, open(data_path + name + "-CV-" + str(iii) + ".p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
#     with open(data_path+name+'-train-CV-'+str(iii)+'.txt', 'w') as file:
#         file.write('1\n') # 1 denotes classification
#         file.write(str(len(train))+'\n')
#         file.write(str(data['n_dim'])+'\n')
#         np.savetxt(file,np.hstack((data['x_train'][train]/10,data['y_train'][train,None])),delimiter=' ',fmt='%.1f '*data['n_dim']+'%d')
#     with open(data_path+name+'-test-CV-'+str(iii)+'.txt', 'w') as file:
#         file.write('1\n') # 1 denotes classification
#         file.write(str(len(test))+'\n')
#         file.write(str(data['n_dim'])+'\n')
#         np.savetxt(file,np.hstack((data['x_train'][test]/10,data['y_train'][test,None])),delimiter=' ',fmt='%.1f '*data['n_dim']+'%d')
#     iii = iii + 1 
