import random as pyrandom
import math
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pickle
import jax.numpy as jnp
import copy
from numpyro.infer.util import log_likelihood
from class_defs import BayesianTree, Node
from scipy.special import expit

# Note true tree id for large-real dataset is '7171811126'
# Note true tree id for cgm dataset is '391056'

################## Command-line related function ################## 
def parser_general_options(parser): 
    general_options = parser.add_argument_group("General Options")
    general_options.add_argument('--dataset', dest='dataset', default='toy-class',
        help='dataset to be used (default: %(default)s)')
    general_options.add_argument('--save', dest='save', default=0, type=int,
        help='do you wish to save the results? (1=Yes/0=No)') 
    general_options.add_argument('--tag', dest='tag', default='', 
            help='additional tag to identify results from a particular run')
    general_options.add_argument('--init_id', dest='init_id', default=1, type=int,
            help='seed value to change initialisation for multiple runs')
    general_options.add_argument('--datapath', dest='datapath', default='',
            help='path to the dataset')
    general_options.add_argument('--out_dir', dest='out_dir', default='.', 
            help='output directory for pickle files (NOTE: make sure directory exists) (default: %(default)s)')
    return parser

def parser_search_options(parser): # options relevant to the search loop (topology moves)
    search_options = parser.add_argument_group("Tree Search Options")
    search_options.add_argument('--T', dest='T', default=500, type=int,
        help='number of search iterations to run local sampling (default: %(default)s)')
    search_options.add_argument('--T0', dest='T0', default=100, type=int,
        help='number of initial trees to generate from prior (default: %(default)s)')
    search_options.add_argument('--C0', dest='C0', default=5, type=int,
        help='threshold for number of times tree is proposed before adding to active trees (default: %(default)s)')
    search_options.add_argument('--Tmax', dest='Tmax', default=100, type=int,
        help='maximum number of active trees to keep track of during serach (default: %(default)s)')
    search_options.add_argument('--move_probs', nargs='+', dest='move_probs', default=[0.6,0.3,0.1], type=float,
            help='probability of suggesting move proposal from current tree topology in order [grow,prune,stay] (default: %(default)s)')
    return parser

def parser_model_options(parser): # options relevant to defining the model (i.e. prior/likelihood hyperparameters)
    model_options = parser.add_argument_group("Model Options")
    model_options.add_argument('--alpha', dest='alpha', default=0.95, type=float,
        help='alpha parameter for prior on tree structure (default: %(default)s)')   
    model_options.add_argument('--beta', dest='beta', default=1, type=float,
        help='beta parameter for prior on tree structure (default: %(default)s)')
    model_options.add_argument('--alpha_llh', dest='alpha_llh', default=[1], type=float,
        help='alpha concentration parameter for Dir-Multi (classification) likelihood (default: %(default)s)')
    model_options.add_argument('--scale_llh', dest='scale_llh', default=3/2, type=float,
        help='inverse-gamma scale parameter for regression likelihood (default: %(default)s)')
    model_options.add_argument('--mu_mean_llh', dest='mu_mean_llh', default=0.0, type=float,
        help='normal distribution mean parameter for regression likelihood (default: %(default)s)')
    model_options.add_argument('--mu_var_llh', dest='mu_var_llh', default=1.0, type=float,
        help='normal distribution variance parameter for regression likelihood (default: %(default)s)')
    return parser

def parser_hmc_options(parser): # options relevant to locally running the HMC sampler
    hmc_options = parser.add_argument_group("HMC Options")
    hmc_options.add_argument('--num_warmup', dest='num_warmup', default=1000, type=int,
        help='number of warm-up samples to burn in HMC for each new tree structure (default: %(default)s)')
    hmc_options.add_argument('--num_samps', dest='num_samps', default=1000, type=int,
        help='number of HMC samples at each subsequent local iteration (default: %(default)s)')
    hmc_options.add_argument('--num_chains', dest='num_chains', default=4, type=int,
        help='number of independent HMC sampler chains to run in parallel (default: %(default)s)')
    hmc_options.add_argument('--prop_density', dest='prop_density', default=0, type=int,
        help='proposal density used in marginal likelihood estimation - basic=0 or per-chain(spatial)=1 (default: %(default)s)')
    hmc_options.add_argument('--M', dest='M', default=100, type=int,
        help='number of importance samples to estimate marginal likelihood via PI-MAIS method (default: %(default)s)')
    hmc_options.add_argument('--h_init', dest='h_init', default=0.1, type=float,
        help='initial value for gating function parameter for soft splits (default: %(default)s)')
    hmc_options.add_argument('--h_final', dest='h_final', default=0.005, type=float,
        help='final value for gating function parameter for soft splits (default: %(default)s)')
    hmc_options.add_argument('--dense_mass', dest='dense_mass', default=False, action='store_true',
        help='include argument in call to use dense mass matrix within HMC adaptation (default: %(default)s)')
    hmc_options.add_argument('--delta', dest='delta', default=0, type=float,
        help='hyperparameter that controls trade off between exploration/exploitation (default: full exploration - %(default)s)')
    hmc_options.add_argument('--kappa', dest='kappa', default=0, type=float,
        help='smoothness parameter to trade-off terms when calculating exploration term - %(default)s)')
    hmc_options.add_argument('--beta_opt', dest='beta_opt', default=1, type=float,
        help='standard optimism boost hyperparameter, must be >0 (default: %(default)s)')
    hmc_options.add_argument('--Ta', dest='Ta', default=1000, type=float,
        help='number of look-ahead samples used to for calculating exploration term (default: %(default)s)') # default value taken from (Rainforth, et al. 2018) paper
    return parser

def process_command_line():
    parser = argparse.ArgumentParser()
    parser = parser_general_options(parser)
    parser = parser_search_options(parser)
    parser = parser_model_options(parser)
    parser = parser_hmc_options(parser)
    args = parser.parse_args()
    return args

################## Dataset related functions ##################
def normalise_data(data):
    data['transform'] = {'min':np.amin(data['x_train'],axis=0), 'range':np.amax(data['x_train'],axis=0) - np.amin(data['x_train'],axis=0)}
    # data['transform'] = {'min':np.amin(data['x_train']), 'range':np.amax(data['x_train']) - np.amin(data['x_train'])}
    data['x_train'] = (data['x_train'] - data['transform']['min']) / data['transform']['range'] 
    data['x_test'] = (data['x_test'] - data['transform']['min']) / data['transform']['range'] 
    return data

def convert_data(data): 
    data['x_train'] = jnp.asarray(data['x_train'])
    data['y_train'] = jnp.asarray(data['y_train'])
    if(data['n_test'] > 0):
        data['x_test'] = jnp.asarray(data['x_test'])
        data['y_test'] = jnp.asarray(data['y_test'])
    if(np.shape(data['x_train'])[0] != data['n_train']):
        data['x_train'] = np.transpose(data['x_train'])
        assert (np.shape(data['x_train'])[0]==data['n_train']) and (np.shape(data['x_train'])[1]==data['nx'])
    return data

def process_dataset(data):
    data = normalise_data(data)
    data = convert_data(data)
    return data

def load_data(settings):
    data = {}
    if settings.dataset == 'toy-class':
        data = load_toy_class_data()
    elif settings.dataset == 'toy-non-sym':
        data = load_toy_non_sym()
    elif settings.dataset == 'toy-class-two-split':
        data = load_basic_toy_class_two_splits_data()
    elif settings.dataset == 'toy-class-noise':
        data = load_toy_class_noise_data()
    else:
        try:
            dt = pickle.load(open(settings.datapath + settings.dataset + '.pickle', "rb"))
        except:
            raise Exception('Unknown dataset: ' + settings.datapath + settings.dataset)
        data = import_external_dataset(dt)
    return data 

def import_external_dataset(dt):
    if('output_type' not in dt):
        raise Exception('Output type is not specified in dataset (must be either class/real for classification/regression).')   
    if(np.shape(dt['x_train'])[0] != dt['n_train']):
        dt['x_train'] = np.transpose(dt['x_train'])
        assert (np.shape(dt['x_train'])[0]==dt['n_train']) and (np.shape(dt['x_train'])[1]==dt['nx'])
    if('x_test' not in dt): # handles case when importing just training data
        dt['x_test'] = []
        dt['y_test'] = []
        dt['n_test'] = 0
        if(np.shape(dt['x_test'])[0]!=dt['n_test']):
            dt['x_test'] = np.transpose(dt['x_test'])
            assert (np.shape(dt['x_test'])[0]==dt['n_test']) and (np.shape(dt['x_test'])[1]==dt['nx'])
    return dt

def load_toy_non_sym():
    """ Toy dataset which is not symmetric. """
    tau1 = 0.5
    tau2 = 0.7 # x(0) > 0.5
    tau3 = 0.3 # x(0) <= 0.5
    indx1 = 0
    indx2 = 1 
    indx3 = 1
    nx = 2 
    ny = 2
    n_train = 1000
    n_test = n_train

    x_train = np.empty((n_train,nx))
    x_train[:500,indx1] = np.random.uniform(0.1,tau1-0.05,500)
    x_train[500:,indx1] = np.random.uniform(tau1+0.05,0.9,500)
    x_train[:250,indx3] = np.random.uniform(0.1,tau3-0.05,250)
    x_train[250:500,indx3] = np.random.uniform(tau3+0.05,0.9,250)
    x_train[500:750,indx2] = np.random.uniform(0.1,tau2-0.05,250)
    x_train[750:,indx2] = np.random.uniform(tau2+0.05,0.9,250)

    x_test = np.empty((n_test,nx))
    x_test[:500,indx1] = np.random.uniform(0.1,tau1-0.05,500)
    x_test[500:,indx1] = np.random.uniform(tau1+0.05,0.9,500)
    x_test[:250,indx3] = np.random.uniform(0.1,tau3-0.05,250)
    x_test[250:500,indx3] = np.random.uniform(tau3+0.05,0.9,250)
    x_test[500:750,indx2] = np.random.uniform(0.1,tau2-0.05,250)
    x_test[750:,indx2] = np.random.uniform(tau2+0.05,0.9,250)

    y_train = np.empty(n_train)
    y_train[(x_train[:,indx1] <= tau1) & (x_train[:,indx3] <= tau3)] =  1 + np.random.normal(0,0.1,250)
    y_train[(x_train[:,indx1] <= tau1) & (x_train[:,indx3] > tau3)] =  0 + np.random.normal(0,0.1,250)
    y_train[(x_train[:,indx1] > tau1) & (x_train[:,indx2] <= tau2)] =  0 + np.random.normal(0,0.1,250)
    y_train[(x_train[:,indx1] > tau1) & (x_train[:,indx2] > tau2)] =  1 + np.random.normal(0,0.1,250)

    y_test = np.empty(n_test)
    y_test[(x_test[:,indx1] <= tau1) & (x_test[:,indx3] <= tau3)] =  1 + np.random.normal(0,0.1,250)
    y_test[(x_test[:,indx1] <= tau1) & (x_test[:,indx3] > tau3)] =  0 + np.random.normal(0,0.1,250)
    y_test[(x_test[:,indx1] > tau1) & (x_test[:,indx2] <= tau2)] =  0 + np.random.normal(0,0.1,250)
    y_test[(x_test[:,indx1] > tau1) & (x_test[:,indx2] > tau2)] =  1 + np.random.normal(0,0.1,250)

    data = {'x_train': x_train, 'y_train': y_train, 'ny': ny, \
            'nx': nx, 'n_train': n_train, 'x_test': x_test, \
            'y_test': y_test, 'n_test': n_test, 'output_type':'real'}
    return data

def load_toy_class_data():
    p = 3 # number of predictors
    n = 1000 # number of training observations
    n_test = 1000 # number of testing observations
    x_train = np.zeros([n,p])
    y_train = np.zeros(n)
    x_test = np.zeros([n_test,p])
    y_test = np.zeros(n_test)

    # # one split
    # eps = 0.0 # deadzone region
    # tau_true = 0.5

    # x_train[:,0] = np.random.uniform(0,1,n) # x0 is random, does not effect output
    # x_train[:,2] = np.random.uniform(0,1,n) # x2 is random, does not effect output
    # indx = np.random.uniform(size=n) < tau_true
    # x_train[indx,1] = np.random.uniform(0,tau_true-eps,size=sum(indx))
    # x_train[~indx,1] = np.random.uniform(tau_true+eps,1,size=n-sum(indx))
    # indx = x_train[:,1] <= tau_true
    # y_train[indx] = 0
    # y_train[~indx] = 1

    # x_test[:,0] = np.random.uniform(0,1,n_test) # x0 is random, does not effect output
    # x_test[:,2] = np.random.uniform(0,1,n_test) # x2 is random, does not effect output
    # indx = np.random.uniform(size=n_test) < tau_true
    # x_test[indx,1] = np.random.uniform(0,tau_true-eps,size=sum(indx))
    # x_test[~indx,1] = np.random.uniform(tau_true+eps,1,size=n_test-sum(indx))
    # indx = x_test[:,1] <= tau_true
    # y_test[indx] = 0
    # y_test[~indx] = 1

    ## two splits
    eps = 0.01 # deadzone region
    tau_true = [0.3,0.7]

    x_train[:,0] = np.random.uniform(0,1,n) # x0 is random, does not effect output
    x_train[:,2] = np.random.uniform(0,1,n) # x2 is random, does not effect output
    indx = np.random.uniform(size=n)
    indx1 = indx < tau_true[0]
    indx2 = (tau_true[0] <= indx) & (indx < tau_true[1])
    indx3 = indx >= tau_true[1]
    x_train[indx1,1] = np.random.uniform(0,tau_true[0]-eps,size=sum(indx1))
    x_train[indx2,1] = np.random.uniform(tau_true[0]+eps,tau_true[1]-eps,size=sum(indx2))
    x_train[indx3,1] = np.random.uniform(tau_true[1]+eps,1,size=sum(indx3))
    y_train[indx1] = 0
    y_train[indx2] = 1
    y_train[indx3] = 0

    x_test[:,0] = np.random.uniform(0,1,n_test) # x0 is random, does not effect output
    x_test[:,2] = np.random.uniform(0,1,n_test) # x2 is random, does not effect output
    indx = np.random.uniform(size=n)
    indx1 = indx < tau_true[0]
    indx2 = (tau_true[0] <= indx) & (indx < tau_true[1])
    indx3 = indx >= tau_true[1]
    x_test[indx1,1] = np.random.uniform(0,tau_true[0]-eps,size=sum(indx1))
    x_test[indx2,1] = np.random.uniform(tau_true[0]+eps,tau_true[1]-eps,size=sum(indx2))
    x_test[indx3,1] = np.random.uniform(tau_true[1]+eps,1,size=sum(indx3))
    y_test[indx1] = 0
    y_test[indx2] = 1
    y_test[indx3] = 0

    data = {'x_train': x_train, 'y_train': y_train, 'ny': 2, \
            'nx': p, 'n_train': n, 'x_test': x_test, 'y_test': y_test, \
            'n_test': n_test, 'output_type':'class'}
    return data

def load_basic_toy_class_data():
    p = 1 # number of predictors
    n = 1000 # number of training observations
    n_test = 1000 # number of testing observations
    x_train = np.zeros([n,p])
    y_train = np.zeros(n)
    x_test = np.zeros([n_test,p])
    y_test = np.zeros(n_test)
    eps = 0.05 # deadzone region
    tau_true = 0.5

    # x_train[:,0] = np.random.uniform(0,1,n) # x0 is random, does not effect output
    # x_train[:,2] = np.random.uniform(0,1,n) # x2 is random, does not effect output
    indx = np.random.uniform(size=n) < tau_true
    x_train[indx,0] = np.random.uniform(0,tau_true-eps,size=sum(indx))
    x_train[~indx,0] = np.random.uniform(tau_true+eps,1,size=n-sum(indx))
    indx = x_train[:,0] <= tau_true
    y_train[indx] = 0
    y_train[~indx] = 1

    # x_test[:,0] = np.random.uniform(0,1,n_test) # x0 is random, does not effect output
    # x_test[:,2] = np.random.uniform(0,1,n_test) # x2 is random, does not effect output
    indx = np.random.uniform(size=n_test) < tau_true
    x_test[indx,0] = np.random.uniform(0,tau_true-eps,size=sum(indx))
    x_test[~indx,0] = np.random.uniform(tau_true+eps,1,size=n_test-sum(indx))
    indx = x_test[:,0] <= tau_true
    y_test[indx] = 0
    y_test[~indx] = 1

    data = {'x_train': x_train, 'y_train': y_train, 'ny': 2, \
            'nx': p, 'n_train': n, 'x_test': x_test, 'y_test': y_test, \
            'n_test': n_test, 'output_type':'class'}
    return data

def load_basic_toy_class_two_splits_data():
    p = 1 # number of predictors
    n = 1000 # number of training observations
    n_test = 1000 # number of testing observations
    x_train = np.zeros([n,p])
    y_train = np.zeros(n)
    x_test = np.zeros([n_test,p])
    y_test = np.zeros(n_test)

    # ## one split
    # eps = 0.05 # deadzone region
    # tau_true = 0.3

    # # x_train[:,0] = np.random.uniform(0,1,n) # x0 is random, does not effect output
    # # x_train[:,2] = np.random.uniform(0,1,n) # x2 is random, does not effect output
    # indx = np.random.uniform(size=n) < tau_true
    # x_train[indx,0] = np.random.uniform(0,tau_true-eps,size=sum(indx))
    # x_train[~indx,0] = np.random.uniform(tau_true+eps,1,size=n-sum(indx))
    # indx = x_train[:,0] <= tau_true
    # y_train[indx] = 0
    # y_train[~indx] = 1

    # # x_test[:,0] = np.random.uniform(0,1,n_test) # x0 is random, does not effect output
    # # x_test[:,2] = np.random.uniform(0,1,n_test) # x2 is random, does not effect output
    # indx = np.random.uniform(size=n_test) < tau_true
    # x_test[indx,0] = np.random.uniform(0,tau_true-eps,size=sum(indx))
    # x_test[~indx,0] = np.random.uniform(tau_true+eps,1,size=n_test-sum(indx))
    # indx = x_test[:,0] <= tau_true
    # y_test[indx] = 0
    # y_test[~indx] = 1

    ## two splits
    eps = 0.01 # deadzone region
    tau_true = [0.3,0.7]

    # x_train[:,0] = np.random.uniform(0,1,n) # x0 is random, does not effect output
    # x_train[:,2] = np.random.uniform(0,1,n) # x2 is random, does not effect output
    indx = np.random.uniform(size=n)
    indx1 = indx < tau_true[0]
    indx2 = (tau_true[0] <= indx) & (indx < tau_true[1])
    indx3 = indx >= tau_true[1]
    x_train[indx1,0] = np.random.uniform(0,tau_true[0]-eps,size=sum(indx1))
    x_train[indx2,0] = np.random.uniform(tau_true[0]+eps,tau_true[1]-eps,size=sum(indx2))
    x_train[indx3,0] = np.random.uniform(tau_true[1]+eps,1,size=sum(indx3))
    y_train[indx1] = 0
    y_train[indx2] = 1
    y_train[indx3] = 0

    # x_test[:,0] = np.random.uniform(0,1,n_test) # x0 is random, does not effect output
    # x_test[:,2] = np.random.uniform(0,1,n_test) # x2 is random, does not effect output
    indx = np.random.uniform(size=n)
    indx1 = indx < tau_true[0]
    indx2 = (tau_true[0] <= indx) & (indx < tau_true[1])
    indx3 = indx >= tau_true[1]
    x_test[indx1,0] = np.random.uniform(0,tau_true[0]-eps,size=sum(indx1))
    x_test[indx2,0] = np.random.uniform(tau_true[0]+eps,tau_true[1]-eps,size=sum(indx2))
    x_test[indx3,0] = np.random.uniform(tau_true[1]+eps,1,size=sum(indx3))
    y_test[indx1] = 0
    y_test[indx2] = 1
    y_test[indx3] = 0

    data = {'x_train': x_train, 'y_train': y_train, 'ny': 2, \
            'nx': p, 'n_train': n, 'x_test': x_test, 'y_test': y_test, \
            'n_test': n_test, 'output_type':'class'}
    return data

def load_toy_class_noise_data():
    data = load_toy_class_data()
    # Create noisy dataset - randomly swap values of some indicies
    indx = pyrandom.sample(range(0,data['n_train']), int(np.floor(data['n_train']/10)))
    data['y_train']= np.array(data['y_train'])
    data['y_train'][indx] = 1 - data['y_train'][indx]
    data['y_train'] = data['y_train']
    indx = pyrandom.sample(range(0,data['n_test']), int(np.floor(data['n_test']/10)))
    data['y_test']= np.array(data['y_test'])
    data['y_test'][indx] = 1 - data['y_test'][indx]
    data['y_test'] = data['y_test']
    return data

def load_toy_class_1d_data():
    """ Create basic 1d dataset - Gaussian distributed x values, y in {0,1}. """
    p = 1 # number of predictors
    n_train = 100 # number of observations
    n_test = n_train
    x_train = np.zeros([n_train,p])
    y_train = np.zeros(n_train)
    x_test = np.zeros([n_test,p])
    y_test = np.zeros(n_test)
    tau_true = 0.5

    indx = np.random.uniform(size=n_train) < tau_true
    x_train[indx] = np.random.normal(0.25,0.08,size=(sum(indx),1))
    y_train[indx] = 0
    x_train[~indx] = np.random.normal(0.75,0.08,size=(n_train-sum(indx),1))
    y_train[~indx] = 1

    indx = np.random.uniform(size=n_test) < tau_true
    x_test[indx] = np.random.normal(0.25,0.08,size=(sum(indx),1))
    y_test[indx] = 0
    x_test[~indx] = np.random.normal(0.75,0.08,size=(n_test-sum(indx),1))
    y_test[~indx] = 1
    data = {'x_train': x_train, 'y_train': y_train, 'ny': 2, \
        'nx': p, 'n_train': n_train, 'x_test': x_test, 'y_test': y_test, \
        'n_test': n_test, 'output_type':'class'}
    return data

def plot_dataset(data, ax=None, plot_now=False):
    if(data['nx'] == 1): # one predictor variable 
        if(ax is None):
            plot_now = True
            plt.figure(figsize=(15,10))  
        plt.plot(data['x_train'],data['y_train'],'*')
        plt.xlabel("x")
        plt.ylabel("y")
    elif(data['nx'] == 2): # two predictor variables
        if(ax is None):
            plot_now = True
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data['x_train'][:,0],data['x_train'][:,1],data['y_train'])
        ax.set_xlabel('x0')
        ax.set_ylabel('x1')
        ax.set_zlabel('y')
    else: # otherwise, visualise separately
        if(ax is None):
            plot_now = True
            fig, ax = plt.subplots(1,data['nx'],figsize=(15,10)) 
        for i in range(data['nx']):
            ax[i].plot(np.array(data['x_train'])[:,i],data['y_train'],'*')
            ax[i].set_xlabel("x"+str(i))
            ax[i].set_ylabel("y")

    if(plot_now == True):
        plt.show()

################## Tree Related Functions ##################
def generate_trees_cgm(T0,data,settings):
    """Generates and returns a set of trees samples from the prior defined in Chipman, et al. 1998.
        Sampling is run for T0 loops but may return less trees due to multiple
    """
    trees = {}

    for i in range(T0):
        tree = BayesianTree(data=data,settings=settings)
        active_nodes = [tree.root_node]
        while(len(active_nodes)>0):
            node = active_nodes.pop(0)
            p_split = settings.alpha * np.power((1+node.get_depth()),-settings.beta)
            if(np.random.rand() < p_split):
                new_left = Node(parent=node,left=None,right=None,ancestors=node.ancestors+[node],path=node.path+[0])
                new_right= Node(parent=node,left=None,right=None,ancestors=node.ancestors+[node],path=node.path+[1])
                node.left = new_left
                node.right = new_right
                new_left.node_id = new_left.get_node_id()
                new_right.node_id = new_right.get_node_id()
                active_nodes.extend([new_left,new_right])

        tree.reset_tree(data)
        tree_id = ''.join(str(node.get_node_id()) for node in tree.leaf_nodes)
        if tree_id in trees.keys():
            trees[tree_id]['times_proposed'] += 1
        else:
            trees[tree_id] = {'T':tree,'times_proposed':1,'times_selected':0,'samples':None,'log_marg_llh':None,'log_w':None,'Ck':None,'tau_tilde':None,'dist_logw':None,'p':None}

    return trees

def get_child_id(node_id,x,split,index,internal_map): # needed to generate large dataset quickly
    tmp = 2 * (node_id + 1)
    if(x[index[internal_map[node_id]]] <= split[internal_map[node_id]]):
        return tmp - 1
    else:
        return tmp

def traverse(x,index,split,leaf_nodes,internal_map):  # needed to generate large dataset quickly
    node_id = 0
    while (node_id not in leaf_nodes):
        node_id = get_child_id(node_id,x,split,index,internal_map)
    return node_id

def traverse_all_data(x,index,tau,internal_nodes,leaf_nodes): # needed to generate large dataset quickly
    node_ids = np.zeros(len(x),dtype=int)
    leaf_map = np.full(max(leaf_nodes)+1,100)
    leaf_map[leaf_nodes] = np.arange(len(leaf_nodes))
    internal_map = np.full(max(internal_nodes)+1,100)
    internal_map[internal_nodes] = np.arange(len(internal_nodes))
    for i in range(len(x)):
        node_ids[i] = leaf_map[traverse(x[i],index,tau,leaf_nodes,internal_map)]
    return node_ids
