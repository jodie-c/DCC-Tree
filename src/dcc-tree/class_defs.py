# Import required packages
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import numpy as np
import numpyro
import numpyro.distributions as dist
import jax
from jax.scipy.special import expit, gammaln
import jax.numpy as jnp

################## Class definitions ##################
# Define node class
class Node(object):
    def __init__(self,parent=None,left=None,right=None,params=None,ancestors=None,path=None):
        self.parent = parent
        if(ancestors is None):
            self.ancestors = []
        else:
            self.ancestors = ancestors
        if(path is None):
            self.path = []
        else:
            self.path = path
        if(left == -1): # handles structure initialisation
            self.left = None
            self.right = None
        else:
            self.left = left
            self.right = right
        if(params is None):
            self.params = {}
        else:
            self.params = params
        self.node_id = self.get_node_id()

    def get_node_id(self):
        if(self.parent is None):
            return 0
        if(self == self.parent.left):
            return 2 * (self.parent.get_node_id() + 1) - 1
        else:
            return 2 * (self.parent.get_node_id() + 1)

    def get_depth(self):
        if(self.parent is None): # no parents
            return 0
        else:
            return (1 + self.parent.get_depth())

    def get_internal_nodes(self,internal_nodes): 
        if(self.left is not None):
            internal_nodes.append(self)
            self.left.get_internal_nodes(internal_nodes)
            if(self.right.left is not None):
                self.right.get_internal_nodes(internal_nodes)
        return internal_nodes

    def get_leaf_nodes(self,leaf_nodes,get_id=False):
        if(self.left is not None): # node has children
            self.left.get_leaf_nodes(leaf_nodes,get_id)
            self.right.get_leaf_nodes(leaf_nodes,get_id)
        else:
            if(get_id):
                leaf_nodes.append(self.node_id)
            else:
                leaf_nodes.append(self)
        return leaf_nodes

    def get_no_grandchildren(self,no_gc_nodes):
        if(self.left is not None):
            if((self.left.left is not None) or (self.right.left is not None)):
                if(self.left.left is not None):
                    self.left.get_no_grandchildren(no_gc_nodes)
                if(self.right.left is not None):
                    self.right.get_no_grandchildren(no_gc_nodes)
            else:
                no_gc_nodes.append(self)
        return no_gc_nodes

# Define generic Bayesian tree class
class BayesianTree(object):
    def __init__(self,data,settings):
        # Create information for tree structure
        self.root_node = Node()
        self.internal_nodes = self.root_node.get_internal_nodes([])
        self.leaf_nodes = self.root_node.get_leaf_nodes([])
        self.n = len(self.internal_nodes)
        self.nl = self.n + 1 # true for binary trees that nl = n + 1
        self.prior = self.init_prior(settings)
        self.output_type = data['output_type']
        if(self.output_type == 'class'):
            self.y_classes = np.unique(data['y_train'])
            # Determine which datapoints are associcated with which output by index
            indicies = jnp.full([data['ny'],data['n_train']],-1)
            for i in range(data['ny']):
                indicies = indicies.at[i,:].set((data['y_train'] == self.y_classes[i]))
            self.indicies = indicies
            self.alpha_llh = jnp.array(settings.alpha_llh*data['ny']) # assuming equal Dir-Multi concentration params
        elif(self.output_type == 'real'):
            self.scale_llh = settings.scale_llh
            self.mu_mean_llh = settings.mu_mean_llh
            self.mu_var_llh = settings.mu_var_llh
        else:
            AssertionError
            print("Data output type must be specified and be either class (classification) or real (regression).")

    def reset_tree(self,data):
        self.internal_nodes = self.root_node.get_internal_nodes([])
        self.n = len(self.internal_nodes)
        self.leaf_nodes = self.root_node.get_leaf_nodes([])
        self.nl = self.n + 1 # always one more leaf node than internal node in binary tree
        self.calc_tree_prior(data['nx'])

    def calc_tree_prior(self,nx):
        log_p = 0
        for node in self.internal_nodes:
            log_p = log_p + np.log(self.prior['alpha']/np.power(1.0+node.get_depth(),self.prior['beta']))
        for node in self.leaf_nodes:
            log_p = log_p + np.log(1-self.prior['alpha']/np.power(1.0+node.get_depth(),self.prior['beta']))

        self.prior['log_p'] = log_p

    def init_prior(self,settings):
        return {'alpha': settings.alpha, 'beta': settings.beta, 'log_p':np.log((1-settings.alpha/np.power(1.0,settings.beta)))} # depth initially 0
    
    def get_internal_nodes(self):
        return self.root_node.get_internal_nodes([])

    def get_leaf_nodes(self):
        return self.root_node.get_leaf_nodes([])

    def get_no_grandchildren(self):
        return self.root_node.get_no_grandchildren([])

    def update_params(self,samples,indx,data):
        index = np.array(samples['index'][indx])
        tau = np.array(samples['tau'][indx])#*data['transform']['range']+data['transform']['min']

        for i,node in enumerate(self.internal_nodes):
            node.params = {} # reset parameters (removes tau/index in originally internal node)
            node.params['index'] = index[i]
            node.params['tau'] = tau[i]

        if(data['output_type'] == 'real'): 
            for i,node in enumerate(self.leaf_nodes):
                node.params = {}
                node.params['mu'] = np.array(samples['mu'][indx][i])
                node.params['sigma'] = np.array(samples['sigma'][indx]) # assuming constant variance
        else:
            for i,node in enumerate(self.leaf_nodes):
                pass
                # node.params['theta'] = np.array(samples['freqs-'+str(i)][indx])

    def model_tree_disc(self,data,h):
        if(data['output_type']=='class'):
            if(data['nx']>1): # more than one input variable
                index = numpyro.sample('index',dist.Categorical(probs=np.array([1/data['nx']]*data['nx'])), sample_shape=(self.n,)) 
            tau = numpyro.sample('tau',dist.Beta(1,1),sample_shape=(self.n,)) 
            for i,node in enumerate(self.internal_nodes): 
                if(data['nx']==1): # only one input variable
                    node.psi = expit((data['x_train']-tau[i])/h)
                else:
                    node.psi = expit((data['x_train'][:,index[i]]-tau[i])/h)
            logllh_total = 0
            for ii, node in enumerate(self.leaf_nodes):
                phi = 1
                for i,eta in enumerate(node.ancestors): 
                    phi = phi * (eta.psi ** (node.path[i])) * ((1-eta.psi) ** (1-node.path[i]))
                freqs = jnp.sum(phi*self.indicies,axis=1)
                # numpyro.deterministic('freqs-'+str(ii),freqs)# causes issue with jax tracer
                logllh_total += jnp.sum(gammaln(freqs+self.alpha_llh)) - gammaln(jnp.sum(freqs)+jnp.sum(self.alpha_llh))
            numpyro.factor('log_prob', logllh_total)
            for node in self.internal_nodes: # clean up
                delattr(node, 'psi')
            return           
        elif(data['output_type']=='real'):
            if(data['nx']>1): # more than one input variable
                index = numpyro.sample('index',dist.Categorical(probs=np.array([1/data['nx']]*data['nx'])), sample_shape=(self.n,))
            tau = numpyro.sample('tau',dist.Beta(1,1),sample_shape=(self.n,)) 
            sigma = numpyro.sample('sigma', dist.InverseGamma(self.scale_llh,self.scale_llh)) # currently assuming constant variance across leaf nodes
            mu = numpyro.sample('mu', dist.Normal(self.mu_mean_llh*jnp.ones(self.nl),self.mu_var_llh))#numpyro.sample('mu', dist.Normal(jnp.zeros(self.nl),1))
            for i,node in enumerate(self.internal_nodes): 
                if(data['nx']==1): # only one input variable
                    node.psi = expit((data['x_train']-tau[i])/h)
                else:
                    node.psi = expit((data['x_train'][:,index[i]]-tau[i])/h)
            leaf_probs = jnp.zeros((self.nl,data['n_train'])) # array to hold g_k(tau,mu) for each datapoint
            for ii, node in enumerate(self.leaf_nodes):
                phi = 1
                for i,eta in enumerate(node.ancestors): 
                    phi = phi * (eta.psi ** (node.path[i])) * ((1-eta.psi) ** (1-node.path[i]))
                leaf_probs = leaf_probs.at[ii,:].set((mu[ii] - data['y_train']) * phi)
            g = jnp.sum(leaf_probs,axis=0)
            logllh_total = -0.5*data['n_train']*jnp.log(2*jnp.pi*sigma*sigma) - (1/(2*sigma*sigma))*jnp.dot(g,g)
            numpyro.factor('log_prob',logllh_total)
            for node in self.internal_nodes: # clean up
                delattr(node, 'psi')
            return           
        else:
            AssertionError
            print("Data output type has not been specified")
            return

    def model_tree_soft(self,data,h):
        if(data['output_type']=='class'):
            if(data['nx']>1): # more than one input variable
                index = numpyro.sample('index',dist.Dirichlet(np.ones((data['nx'],))),sample_shape=(self.n,)) 
            tau = numpyro.sample('tau',dist.Beta(1,1),sample_shape=(self.n,))
            for i,node in enumerate(self.internal_nodes): 
                if(data['nx']==1): # only one input variable
                    node.psi = expit((data['x_train']-tau[i])/h).T
                else:
                    node.psi = expit((jnp.dot(data['x_train'],index[i,:])-tau[i])/h)
            logllh_total = 0
            for ii, node in enumerate(self.leaf_nodes):
                phi = 1#jnp.ones((data['n_train'],1))
                for i,eta in enumerate(node.ancestors): 
                    phi = phi * (eta.psi ** (node.path[i])) * ((1-eta.psi) ** (1-node.path[i]))
                freqs = jnp.sum(phi*self.indicies,axis=1)
                # numpyro.deterministic('freqs-'+str(ii),freqs) # causes issue with jax tracer
                logllh_total += jnp.sum(gammaln(freqs+self.alpha_llh)) - gammaln(jnp.sum(freqs)+jnp.sum(self.alpha_llh))
            numpyro.factor('log_prob', logllh_total)
            for node in self.internal_nodes: # clean up
                delattr(node, 'psi')
            return
        elif(data['output_type']=='real'):
            if(data['nx']>1): # more than one input variable
                index = numpyro.sample('index',dist.Dirichlet(np.ones((data['nx'],))),sample_shape=(self.n,))
            tau = numpyro.sample('tau',dist.Beta(1,1),sample_shape=(self.n,))
            sigma = numpyro.sample('sigma', dist.InverseGamma(self.scale_llh,self.scale_llh)) # currently assuming constant variance across leaf nodes
            mu = numpyro.sample('mu', dist.Normal(self.mu_mean_llh*jnp.ones(self.nl),self.mu_var_llh))#numpyro.sample('mu', dist.Normal(jnp.zeros(self.nl),1))
            for i,node in enumerate(self.internal_nodes): 
                if(data['nx']==1): # only one input variable
                    node.psi = expit((data['x_train']-tau[i])/h)
                else:
                    node.psi = expit((jnp.dot(data['x_train'],index[i,:])-tau[i])/h)
            leaf_probs = jnp.zeros((self.nl,data['n_train'])) # array to hold g_k(tau,mu) for each datapoint
            for ii, node in enumerate(self.leaf_nodes):
                phi = 1
                for i,eta in enumerate(node.ancestors): 
                    phi = phi * (eta.psi ** (node.path[i])) * ((1-eta.psi) ** (1-node.path[i]))
                leaf_probs = leaf_probs.at[ii,:].set((mu[ii] - data['y_train']) * phi)
            g = jnp.sum(leaf_probs,axis=0)
            logllh_total = -0.5*data['n_train']*jnp.log(2*jnp.pi*sigma*sigma) - (1/(2*sigma*sigma))*jnp.dot(g,g)
            numpyro.factor('log_prob',logllh_total)
            for node in self.internal_nodes: # clean up
                delattr(node, 'psi')
            return
        else:
            AssertionError
            print("Data output type has not been specified")
            return

    def print_tree(self,node):
        print("**********************")
        if(node.parent is not None):
            print("Parent: ",node.parent.node_id,node.parent)
        print("Node: ",node.node_id,node)
        print(node.params)
        if(node.left is not None): # node has children
            print("Left, Right: ",node.left.node_id,node.left,node.right.node_id,node.right)
            self.print_tree(node.left)
            self.print_tree(node.right)

    def draw_tree(self,ax=None):
        treeGraph = nx.DiGraph()
        label_dict = {}
        for node in self.internal_nodes:
            # label_dict[str(node.node_id)] = "x"+str(np.argmax(node.params['index']))+"<"+"{:.2f}".format(node.params['tau'])
            treeGraph.add_edge(str(node.node_id),str(node.left.node_id))
            treeGraph.add_edge(str(node.node_id),str(node.right.node_id))
        # for node in self.leaf_nodes:
        #     if(self.output_type == 'real'):
        #         label_dict[str(node.node_id)] = "mu="+"{:.2f}".format(node.params['mu'])
        #     else:
        #         pass
        #         # label_dict[str(node.node_id)] = ["{:.1f}".format(theta) for theta in node.params['theta']] # from jax tracer issue

        if(ax is None):
            plt.figure()
            ax = plt.gca()
        pos = graphviz_layout(treeGraph, prog="dot")

        nx.draw(treeGraph, pos=pos, ax=ax, labels=label_dict, with_labels=True, node_size=50, font_size=6)

# Define class to hold information about the MCMC algorithm
class MCMC_Info(object):
    def __init__(self, h_info={}, num_warmup=1000, num_samps=10000, num_chains=4, T=100, T0=20, C0=5, \
                    Tmax=100, move_probs=[0.4,0.4,0.2], filename=''):
        self.h_info = h_info                        # specifies transition of "sharpness" of splitting/gating function
        self.num_warmup = num_warmup                # number of warm up (burn-in) samples in HMC step
        self.num_samps = num_samps                  # number of HMC samples to evaluate before making new proposal 
        self.num_chains = num_chains                # number of HMC samples to run in parallel as proposals for SMC step
        self.T = T                                 # number of local search iterations
        self.T0 = T0                                # number of initial trees to generate from prior
        self.C0 = C0                                # number of proposals before adding tree to active trees
        self.Tmax  = Tmax                           # maximum number of active trees to keep track of during search
        self.move_probs = move_probs                 # probability of suggesting move proposal - [p_grow,p_prune,p_stay]
        self.filename = filename                    # name of file where data is saved
        self.logw_max = -np.Inf                     # current maximum log weight sample across all tree samples
        self.recalculate_p = False                  # whether the terms needed to calculate exploration term need to be recalculated

