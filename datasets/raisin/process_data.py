# creates compatible dataset formats for multiple methods for comparison

import numpy as np
import pandas as pd
import pickle as pickle
import random
random.seed(1) 

name = 'raisin'
data_path = './'
train_frac = 0.7

data = {}
filename = data_path + name + '.csv'
df = pd.read_csv(filename)
df['Class'] = df['Class'].astype('category').cat.codes
x = df.drop(labels='Class',axis=1).values
y = df['Class'].values
n = x.shape[0]
print('ndim = ',x.shape[1])

data['nx'] = x.shape[1] 
idx = [i for i in range(n)]
random.shuffle(idx)

n_train = int(train_frac * n)
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
pickle.dump(data, open(data_path + name + ".pickle", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

# Make dataset with format compatible with SMC/MCMC method
data.pop('nx', None)
data.pop('ny', None)
data.pop('output_type',None)
data['n_class'] = len(np.unique(y)) 
data['n_dim'] = x.shape[1] 
data['is_sparse'] = False
pickle.dump(data, open(data_path + name + ".p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

# Make dataset with format compatible with Wu method
# NOTE: needs to be scaled before running script -- divided by 10 here
xmax = np.max(data['x_train'],axis=0)
xmin = np.min(data['x_train'],axis=0)
with open(data_path+name+'-train.txt', 'w') as file:
    file.write('1\n') # 1 denotes classification
    file.write(str(n_train)+'\n')
    file.write(str(data['n_dim'])+'\n')
    np.savetxt(file,np.hstack(((data['x_train']-xmin)/(xmax-xmin),data['y_train'][:,None])),delimiter=' ',fmt='%.2f '*data['n_dim']+'%d')

# NOTE: needs to be scaled before running script
with open(data_path+name+'-test.txt', 'w') as file:
    file.write('1\n') # 1 denotes classification
    file.write(str(len(idx_test))+'\n')
    file.write(str(data['n_dim'])+'\n')
    np.savetxt(file,np.hstack(((data['x_test']-xmin)/(xmax-xmin),data['y_test'][:,None])),delimiter=' ',fmt='%.2f '*data['n_dim']+'%d')

# ## Uncomment to create external CV datasets for other methods
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
    pickle.dump(data_tmp, open(data_path + name + "-CV-" + str(iii) + ".p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    xmax = np.max(data_tmp['x_train'],axis=0)
    xmin = np.min(data_tmp['x_train'],axis=0)
    with open('./raisin-train-CV-'+str(iii)+'.txt', 'w') as file:
        file.write('1\n') # 1 denotes classification
        file.write(str(data_tmp['n_train'])+'\n')
        file.write(str(data_tmp['n_dim'])+'\n')
        np.savetxt(file,np.hstack(((data_tmp['x_train']-xmin)/(xmax-xmin),data_tmp['y_train'][:,None])),delimiter=' ',fmt='%.2f '*data_tmp['n_dim']+'%d')
    with open('./raisin-test-CV-'+str(iii)+'.txt', 'w') as file:
        file.write('1\n') # 1 denotes classification
        file.write(str(data_tmp['n_test'])+'\n')
        file.write(str(data_tmp['n_dim'])+'\n')
        np.savetxt(file,np.hstack(((data_tmp['x_test']-xmin)/(xmax-xmin),data_tmp['y_test'][:,None])),delimiter=' ',fmt='%.2f '*data_tmp['n_dim']+'%d')
    iii = iii + 1 