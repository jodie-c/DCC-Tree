import os
import sys
import pickle
import scipy.io
import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.stats import norm
from jax.scipy.special import expit, logsumexp
import numpyro.distributions as dist

#### Uncomment dataset to be considered - NOTE code assumes regression dataset
dataset = "cgm"
# dataset = "wu"

# # For nice plots
# tex_fonts = {
#     "text.usetex": True,
#     "font.family": "serif",
#     "axes.labelsize": 24,
#     "font.size": 24,
#     "legend.fontsize": 18,
#     "xtick.labelsize": 18,
#     "ytick.labelsize": 18
# }

# plt.rcParams.update(tex_fonts)

sys.path.append('../hmc-tree/src/hmc-df/') # for hmc related classes 

def predict_df(tree,x):
    assert(tree['I'][0].get_node_id() == 0)
    
    node = tree['I'][0] # root node
    while(node.left is not None):
        if(x[node.params['index']] < node.params['tau']):
            node = node.left
        else:
            node = node.right
    return node.params['mu'], node.params['sigma']

def predict_dfi(tree,x,h):
    assert(tree['I'][0].get_node_id() == 0)
    nl = len(tree['L'])
    
    for i,node in enumerate(tree['I']): 
        if(data['nx']==1): # only one input variable
            node.psi_test = expit((x-node.params['tau'])/h)
        else:
            node.psi_test = expit((np.dot(x,node.params['index'])-node.params['tau'])/h)
    phi = np.zeros(nl) 
    mu = np.zeros(nl)
    for ii, node in enumerate(tree['L']):
        mu[ii] = node.params['mu']
        phi_test = 1
        for i,eta in enumerate(node.ancestors): 
            phi_test = phi_test * (eta.psi_test ** (node.path[i])) * ((1-eta.psi_test) ** (1-node.path[i]))
        phi[ii] = phi_test
    return mu[np.argmax(phi)], node.params['sigma'] # constant variance assumption

def predict_dfi_all(tree,x_all,h):
    assert(tree['I'][0].get_node_id() == 0)
    nl = len(tree['L'])
    
    for i,node in enumerate(tree['I']): 
        if(data['nx']==1): # only one input variable
            node.psi_test = expit((x_all-node.params['tau'])/h)
        else:
            node.psi_test = expit((np.dot(x_all,node.params['index'])-node.params['tau'])/h)
    phi = np.zeros((len(x_all),nl))
    mu = np.zeros(nl)
    for ii, node in enumerate(tree['L']):
        mu[ii] = node.params['mu']
        phi_test = 1
        for i,eta in enumerate(node.ancestors): 
            phi_test = phi_test * (eta.psi_test ** (node.path[i])) * ((1-eta.psi_test) ** (1-node.path[i]))
        phi[:,ii] = phi_test
    return mu[np.argmax(phi,axis=1)], node.params['sigma'] # constant variance assumption

def predict_dcc(tree, x, h): 
    # NOTE this is different to other methods as it computes one datapoint w.r.t many tree samples all with the same structure
    assert(tree['T'].internal_nodes[0].get_node_id() == 0)
    nl = len(tree['T'].leaf_nodes)
    (M,N_samps,n) = tree['proposals']['tau'].shape

    # Reshape proposal samples and weights
    tau = np.reshape(tree['proposals']['tau'],(M*N_samps,n))
    index = np.reshape(tree['proposals']['index'],(M*N_samps,n,data['nx']))
    mu = np.reshape(tree['proposals']['mu'],(M*N_samps,nl))
    sigma = np.reshape(tree['proposals']['sigma'],(M*N_samps))
    logw_tilde = np.reshape(tree['log_w'],(M*N_samps))

    for i,node in enumerate(tree['T'].internal_nodes): 
        if(data['nx']==1): # only one input variable
            node.psi_test = expit((x-tau[:,i])/h)
        else:
            node.psi_test = expit((np.dot(index[:,i,:],x)-tau[:,i])/h)
    phi = np.zeros((M*N_samps,nl))
    for ii, node in enumerate(tree['T'].leaf_nodes):
        phi_test = 1
        for i,eta in enumerate(node.ancestors): 
            phi_test = phi_test * (eta.psi_test ** (node.path[i])) * ((1-eta.psi_test) ** (1-node.path[i]))
        phi[:,ii] = phi_test
    return mu[np.arange(M*N_samps),np.argmax(phi,axis=1)], sigma, logw_tilde

def normalise_data(data):
    data['transform'] = {'min':np.amin(data['x_train'],axis=0), 'range':np.amax(data['x_train'],axis=0) - np.amin(data['x_train'],axis=0)}
    data['x_train'] = (data['x_train'] - data['transform']['min']) / data['transform']['range'] 
    data['x_test'] = (data['x_test'] - data['transform']['min']) / data['transform']['range'] 
    return data

# Load in metric information
hmc_df = scipy.io.loadmat('./matlab/hmc_data-df-'+dataset+'.mat') 
hmc_dfi = scipy.io.loadmat('./matlab/hmc_data-dfi-'+dataset+'.mat') 

# Load in sample information 
hmc_df['samps'] = {}
hmc_dfi['samps'] = {}
files = os.listdir("../soft-trees/results/final/"+dataset)
for file in files:
    if file.endswith(".pickle") and ("-df-" in file):
        with open("../soft-trees/results/final/"+dataset+"/"+file, "rb") as input_file:
            hmc_data = pickle.load(input_file)
        hmc_df['samps'][str(hmc_data['settings'].init_id)] = hmc_data['samples']
for file in files:
    if file.endswith(".pickle") and ("-dfi-" in file):
        with open("../soft-trees/results/final/"+dataset+"/"+file, "rb") as input_file:
            hmc_data = pickle.load(input_file)
        hmc_dfi['samps'][str(hmc_data['settings'].init_id)] = hmc_data['samples']
files = os.listdir("./results/"+dataset)
for file in files:
    if file.endswith(".pickle"):
        with open("./results/"+dataset+"/"+file, "rb") as input_file:
            dcc_data = pickle.load(input_file)

# Extract information for HMC methods
N_c = len(hmc_df['samps'])     # number of chains run
N = hmc_data['settings'].N      # number of sample iterations (after burn-in)
N_warmup = hmc_data['settings'].N_warmup

# Load in test/train data
data = pickle.load(open("./datasets/"+dataset+"/"+dataset+".pickle", "rb"))

# Transform data to unit interval for H-DFI predictions
data_norm = copy.deepcopy(data)
data_norm = normalise_data(data_norm)

predict_dfi_all(hmc_dfi['samps'][str(1)][str(0)],data_norm['x_test'],hmc_dfi['h_final'])

# Select random testing datapoint to analyse
for ii in range(data['n_test']):
    print("Index of testing datapoint = ", ii)
    print("Test point x = ",data['x_test'][ii],", normalised x = ",data_norm['x_test'][ii],"y = ",data['y_test'][ii])

    y_pred_df = np.zeros((N_c,N))
    y_pred_dfi = np.zeros((N_c,N))

    x_test = data['x_test'][ii]

    x = np.linspace(-5, 15, 1000)
    pdf_sum_df = np.zeros_like(x)
    pdf_sum_dfi = np.zeros_like(x)
    pdf_sum_dcc = np.zeros_like(x)
    
    for i in range(N_c):
        for j in range(N_warmup,N+N_warmup):
            y_pred_df[i,j-N_warmup],sig_df = predict_df(hmc_df['samps'][str(i+1)][str(j)],data['x_test'][ii])
            y_pred_dfi[i,j-N_warmup],sig_dfi = predict_dfi(hmc_dfi['samps'][str(i+1)][str(j)],data_norm['x_test'][ii],hmc_dfi['h_final'])
            pdf_sum_df += norm.pdf(x, loc=y_pred_df[i,j-N_warmup], scale=np.sqrt(sig_df))
            pdf_sum_dfi += norm.pdf(x, loc=y_pred_dfi[i,j-N_warmup], scale=np.sqrt(sig_dfi))

    pdf_sum_df = pdf_sum_df/(N_c*N)
    pdf_sum_dfi = pdf_sum_dfi/(N_c*N)

    logsum_Zm = logsumexp(np.array([dcc_data['active_trees'][tree]['log_marg_llh'] for tree in dcc_data['active_trees']]))
    pdf_sum_dcc_tmp = np.zeros((len(x),len(dcc_data['active_trees'])))
    for i,tree in enumerate(dcc_data['active_trees']):
        mu_dcc, sig_dcc, logw_tilde = predict_dcc(dcc_data['active_trees'][tree],data_norm['x_test'][ii],dcc_data['settings'].h_final)
        pdf_sum_dcc_tmp[:,i] = logsumexp((logw_tilde-logsumexp(logw_tilde))+dist.Normal(mu_dcc,np.sqrt(sig_dcc)).log_prob(np.transpose(x[None])),axis=1)+dcc_data['active_trees'][tree]['log_marg_llh']-logsum_Zm
    pdf_sum_dcc = logsumexp(pdf_sum_dcc_tmp,axis=1)

    plt.figure()
    plt.plot(x, pdf_sum_df)
    plt.plot(x, pdf_sum_dfi)
    plt.plot(x, np.exp(pdf_sum_dcc))
    ylim = plt.ylim()
    plt.vlines(data['y_test'][ii],ylim[0],ylim[1],color='red',linestyle='dashed',label="True Output")
    plt.legend(["HMC-DF Pred Dist","HMC-DFI Pred Dist","DCC-Tree Pred Dist","True Output"])
    plt.ylim(ylim)
    plt.savefig("./results/"+dataset+"/id-"+str(ii)+".pdf")








