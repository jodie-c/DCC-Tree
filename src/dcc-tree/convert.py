import os
import sys
import numpy as np
import pickle
import scipy.io
from jax.scipy.special import expit, logsumexp

#### Uncomment dataset to convert
# dataset = "cgm"
# dataset = "wu"
dataset = "iris"
# dataset = "breast-cancer-wisconsin"
# dataset = "raisin"
# dataset = "wine"
# dataset = "toy-non-sym"
# dataset = "toy-class-basic"


data_info = pickle.load(open("./datasets/"+dataset+"/"+dataset+".pickle","rb"))

##### DCC-Tree Method #####
files = os.listdir("./results/"+dataset+"/")
dcc_acc_train = []
dcc_acc_test = []

def normalise_data(data):
    data['transform'] = {'min':np.amin(data['x_train'],axis=0), 'range':np.amax(data['x_train'],axis=0) - np.amin(data['x_train'],axis=0)}
    data['x_train'] = (data['x_train'] - data['transform']['min']) / data['transform']['range'] 
    data['x_test'] = (data['x_test'] - data['transform']['min']) / data['transform']['range'] 
    return data

def predict_dcc_reg(trees, data, h):
    pred_y_train = np.zeros(data['n_train'])
    pred_y_test = np.zeros(data['n_test'])
    logsum_Zm = logsumexp(np.array([trees[tree]['log_marg_llh'] for tree in trees]))
    for tree in trees:
        nl = len(trees[tree]['T'].leaf_nodes)
        (M,N_samps,n) = trees[tree]['proposals']['tau'].shape

        # Reshape proposal samples and weights
        tau = np.reshape(trees[tree]['proposals']['tau'],(M*N_samps,n))
        index = np.reshape(trees[tree]['proposals']['index'],(M*N_samps,n,data['nx']))
        mu = np.reshape(trees[tree]['proposals']['mu'],(M*N_samps,nl))
        sigma = np.reshape(trees[tree]['proposals']['sigma'],(M*N_samps))
        logw_tilde = np.reshape(trees[tree]['log_w'],(M*N_samps))
        logw = logw_tilde - logsumexp(logw_tilde)

        for i,node in enumerate(trees[tree]['T'].internal_nodes): 
            if(data['nx']==1): # only one input variable
                node.psi_train = expit(np.transpose(data['x_train']-tau[:,i])/h)
                node.psi_test = expit(np.transpose(data['x_test']-tau[:,i])/h)
            else:
                node.psi_train = expit((np.dot(index[:,i,:],np.transpose(data['x_train']))-tau[:,i,None])/h)
                node.psi_test = expit((np.dot(index[:,i,:],np.transpose(data['x_test']))-tau[:,i,None])/h)
        phi_test = np.zeros((M*N_samps,nl,data['n_test']))
        phi_train = np.zeros((M*N_samps,nl,data['n_train']))
        for ii, node in enumerate(trees[tree]['T'].leaf_nodes):
            tmp_train = 1
            tmp_test = 1
            for i,eta in enumerate(node.ancestors): 
                tmp_train = tmp_train * (eta.psi_train ** (node.path[i])) * ((1-eta.psi_train) ** (1-node.path[i]))
                tmp_test = tmp_test * (eta.psi_test ** (node.path[i])) * ((1-eta.psi_test) ** (1-node.path[i]))
            phi_test[:,ii,:] = tmp_test
            phi_train[:,ii,:] = tmp_train
        pred_y_train += np.sum(np.exp(trees[tree]['log_marg_llh']-logsum_Zm)*np.exp(logw)[:,None]*np.sum((mu[:,:,None]*phi_train),axis=1),axis=0)
        pred_y_test += np.sum(np.exp(trees[tree]['log_marg_llh']-logsum_Zm)*np.exp(logw)[:,None]*np.sum((mu[:,:,None]*phi_test),axis=1),axis=0)
        #clean up 
        for node in trees[tree]['T'].internal_nodes:
            del node.psi_train
            del node.psi_test
    return pred_y_train, pred_y_test

def predict_dcc_class(trees, data, h,i):
    pred_y_train = np.zeros((data['n_train'],data['ny']))
    pred_y_test = np.zeros((data['n_test'],data['ny']))
    logsum_Zm = logsumexp(np.array([trees[tree]['log_marg_llh'] for tree in trees]))
    results = {}
    for tree in trees:
        nl = len(trees[tree]['T'].leaf_nodes)
        (M,N_samps,n) = trees[tree]['proposals']['tau'].shape

        # Reshape proposal samples and weights
        tau = np.reshape(trees[tree]['proposals']['tau'],(M*N_samps,n))
        index = np.reshape(trees[tree]['proposals']['index'],(M*N_samps,n,data['nx']))
        logw_tilde = np.reshape(trees[tree]['log_w'],(M*N_samps))
        logw = logw_tilde - logsumexp(logw_tilde)

        for i,node in enumerate(trees[tree]['T'].internal_nodes): 
            if(data['nx']==1): # only one input variable
                node.psi_train = expit(np.transpose(data['x_train']-tau[:,i])/h)
                node.psi_test = expit(np.transpose(data['x_test']-tau[:,i])/h)
            else:
                node.psi_train = expit((np.dot(index[:,i,:],np.transpose(data['x_train']))-tau[:,i,None])/h)
                node.psi_test = expit((np.dot(index[:,i,:],np.transpose(data['x_test']))-tau[:,i,None])/h)

        phi_test = np.zeros((M*N_samps,nl,data['n_test']))
        phi_train = np.zeros((M*N_samps,nl,data['n_train']))
        freqs = np.zeros((M*N_samps,nl,data['ny'])) # holds class frequencies in each leaf node
        freqs_data = np.sum(trees[tree]['T'].indicies,axis=1) # frequency of each output in the training data

        for ii, node in enumerate(trees[tree]['T'].leaf_nodes):
            tmp_train = 1
            tmp_test = 1
            for i,eta in enumerate(node.ancestors): 
                tmp_train = tmp_train * (eta.psi_train ** (node.path[i])) * ((1-eta.psi_train) ** (1-node.path[i]))
                tmp_test = tmp_test * (eta.psi_test ** (node.path[i])) * ((1-eta.psi_test) ** (1-node.path[i]))
            phi_test[:,ii,:] = tmp_test
            phi_train[:,ii,:] = tmp_train
            freqs[:,ii,:] = np.sum(phi_train[:,ii,:,None]*np.transpose(trees[tree]['T'].indicies),axis=1)/freqs_data

        pred_y_train += np.sum(np.exp(trees[tree]['log_marg_llh']-logsum_Zm)*np.exp(logw)[:,None,None]*np.sum((freqs[:,:,None,:]*phi_train[:,:,:,None]),axis=1),axis=0)
        pred_y_test += np.sum(np.exp(trees[tree]['log_marg_llh']-logsum_Zm)*np.exp(logw)[:,None,None]*np.sum((freqs[:,:,None,:]*phi_test[:,:,:,None]),axis=1),axis=0)

        results[tree] = {}
        results[tree]['weights'] = np.exp(trees[tree]['log_marg_llh']-logsum_Zm)*np.exp(logw)[:,None,None]
        results[tree]['pred_y_train'] = np.sum((freqs[:,:,None,:]*phi_train[:,:,:,None]),axis=1)
        results[tree]['pred_y_test'] = np.sum((freqs[:,:,None,:]*phi_test[:,:,:,None]),axis=1)

        #clean up 
        for node in trees[tree]['T'].internal_nodes:
            del node.psi_train
            del node.psi_test

    pickle.dump(results, open('./dcc_data-'+dataset+'-id-'+str(i)+'-info.pickle', "wb"), protocol=pickle.HIGHEST_PROTOCOL) 

    del results   

    return np.argmax(pred_y_train,axis=1), np.argmax(pred_y_test,axis=1)

data_norm = normalise_data(data_info)

results = {}
i=0
for file in files:
    if file.endswith(".pickle") and not file.startswith('.') and ('dfi' not in file): 
        with open("./results/"+dataset+"/"+file, "rb") as input_file:
            dcc_data = pickle.load(input_file)
        if(data_norm['output_type']=='class'):
            pred_y_train, pred_y_test = predict_dcc_class(dcc_data['active_trees'],data_norm,dcc_data['settings'].h_final,i)
            dcc_acc_train.append((1/data_norm['n_train'])*np.sum(data_norm['y_train'] == pred_y_train)) 
            dcc_acc_test.append((1/data_norm['n_test'])*np.sum(data_norm['y_test'] == pred_y_test))
        else:
            pred_y_train, pred_y_test = predict_dcc_reg(dcc_data['active_trees'],data_norm,dcc_data['settings'].h_final)
            dcc_acc_train.append((1/data_norm['n_train'])*np.sum(np.power(pred_y_train-data_norm['y_train'],2)))
            dcc_acc_test.append((1/data_norm['n_test'])*np.sum(np.power(pred_y_test-data_norm['y_test'],2))) 
        i+=1

scipy.io.savemat('./dcc_data-'+dataset+'.mat', mdict={'alpha': dcc_data['settings'].alpha, 'beta': dcc_data['settings'].beta, \
    'h_init': dcc_data['settings'].h_init, 'h_final': dcc_data['settings'].h_final, 
    'dcc_acc_train':dcc_acc_train,'dcc_acc_test': dcc_acc_test})



