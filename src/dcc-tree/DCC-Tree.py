# Algorithm based on Divide, Conquer, Combine paper - notes made throughout code refer back to this paper
# This scripts generates a set of samples and weights from which each subspace distribution can be computed,
# as well as the overall parameter distribution.
# Metrics are computed via a separate script after samples have been generated.

# Import required packages
import time
import random as pyrandom
import matplotlib.pyplot as plt
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import log_prob
from numpyro.distributions import MultivariateNormal
from numpyro.distributions.transforms import StickBreakingTransform, SigmoidTransform, ExpTransform
from jax import random, lax
from jax.scipy.special import logsumexp
from jax.scipy.stats import norm
from utils import *
from class_defs import *

ii = 0 # RNG key

################## Main Function ##################
def main():
    global ii
    settings = process_command_line()

    ii = settings.init_id * 1000

    fname = settings.dataset + "-init_id-" + str(settings.init_id) \
        + "-h_i-" + str(settings.h_init) + "-h_f-" + str(settings.h_final) \
        + "-alpha-" + str(settings.alpha) + "-beta-" + str(settings.beta) \
        + "-delta-" + str(settings.delta) + "-beta_opt-" + str(settings.beta_opt) \
        + "-num_warm-" + str(settings.num_warmup) + "-num_samps-" + str(settings.num_samps) \
        + "-T-" + str(settings.T) + "-T0-" + str(settings.T0) \
        + "-Tmax-" + str(settings.Tmax) + "-C0-" + str(settings.C0) \
        + "-M-" + str(settings.M) + "-tag-" + str(settings.tag) + "-results.pickle"

    # Resetting random seed
    np.random.seed(settings.init_id * 1000)
    pyrandom.seed(settings.init_id * 1000)

    data = load_data(settings)
    print("\nSettings:")
    print(vars(settings))
    numpyro.set_host_device_count(settings.num_chains)
    # plot_dataset(data)

    data = process_dataset(data) # converts to jnp arrays; scales input variables to be on interval [0,1] --> required for global proposals

    # Set up sampling information for the inner MCMC loop methods
    info = MCMC_Info(h_info = {'h_init':settings.h_init,'h_final':settings.h_final},
                    num_warmup = settings.num_warmup, # how many samples to initialise samplers
                    num_samps = settings.num_samps, # how many samples to take at each topology after initialisation
                    num_chains = settings.num_chains, # how many independent samplers to run at each topology - denoted N in original paper Section 6.1
                    T = settings.T,                                 # number of local search iterations
                    T0 = settings.T0,                                # number of initial trees to generate from prior
                    C0 = settings.C0,                                # number of proposals before adding tree to active trees
                    Tmax  = settings.Tmax,                           # maximum number of active trees to keep track of during search
                    move_probs = settings.move_probs,                 # probability of suggesting move proposal - [p_grow,p_prune,p_stay]
                    filename = fname)

    # Create number of tree topologies to explore - note this is specified as a final number of internal nodes where growth is randomly selected
    # Generate initial set of trees from prior - note can implement different priors from literature here to see which is better
    # We currently use the standard CGM prior here
    # Generates trees and saves structure that includes utility information, current samples, current marginal likelihood,
    # number of times proposed
    trees = generate_trees_cgm(info.T0,data,settings)

    try:
        trees.pop('0') 
    except:
        pass

    active_trees = {}

    # --------------------------- Run DCC Tree Algorithm --------------------------- #
    time_start = time.perf_counter()
    for i in range(info.T): # estimate local density (denoted pi_k in paper) from samples via HMC method 
        print("\n----- Iteration ",i," -----\n")

        # Select trees which have been proposed more than C0 times -- Line 3 Alg. 2 DCC Appendix
        ids_to_burn_in = []
        for tree_id in trees:
            if(trees[tree_id]['times_proposed'] >= info.C0):
                active_trees[tree_id] = trees[tree_id]
                if(trees[tree_id]['samples'] is None): # check if any new trees and burn-in -- Line 4 Alg. 2 DCC Appendix
                    ids_to_burn_in.append(tree_id)
                elif(trees[tree_id]['Ck'] is None): # check if this is non-active tree that has been sampled from - only calculate extra info
                    # Calculate log(mean(log(w)))
                    print("tree that was non active selected: ", tree_id)
                    log_marg_llh_est = trees[tree_id]['log_marg_llh']
                    logw = trees[tree_id]['log_w']
                    num_samps = trees[tree_id]['samples']['tau'].shape[0]
                    logw_reshaped = jnp.reshape(logw,(settings.M*num_samps))
                    n = len(logw_reshaped)
                    log_mu = -jnp.log(n) + logsumexp(logw_reshaped)

                    # Calculate log(sigma_k)
                    tmp = np.vstack([logw_reshaped,np.repeat(log_mu,n)])
                    tmp2 = logsumexp(tmp,b=jnp.array([[1],[-1]]),axis=0,return_sign=True)[0]
                    log_sigma_k = -jnp.log(n) + logsumexp(2*tmp2)

                    # Calculate exploitation term via variance of these weights
                    Ck = jnp.max(jnp.array([2*log_marg_llh_est,log_sigma_k]))
                    tau_k_tilde = jnp.sqrt(jnp.exp(2*log_marg_llh_est-Ck) + (1+settings.kappa)*jnp.exp(log_sigma_k-Ck))

                    # Calculate exploration term
                    # First check if any new logw value is larger than then the current global max(logw)
                    logw_max = jnp.max(logw)
                    if(logw_max > info.logw_max):
                        info.logw_max = logw_max
                        info.recalculate_p = True

                    # Calculate hyperparameters of logw density - assuming normal distribution
                    logw_mu = jnp.mean(logw)
                    logw_std = jnp.std(logw)
                    dist_logw = {'mu':logw_mu,'std':logw_std}
                    p_k = 1 - jnp.power(norm.cdf(info.logw_max,logw_mu,logw_std),settings.Ta)
                    trees[tree_id]['Ck'] = Ck
                    trees[tree_id]['tau_tilde'] = tau_k_tilde
                    trees[tree_id]['p'] = p_k
                    trees[tree_id]['dist_logw'] = dist_logw

                    # Check in case chain needs to be restarted
                    if(np.isnan(trees[tree_id]['log_marg_llh'])): 
                        print("log_marg_llh_est is nan -- restarting chains")
                        trees[tree_id] = restart_chains(trees[tree_id],data,settings,info)

        # Burn in new trees
        for id in ids_to_burn_in:
            print('burn-in tree id: ', id)
            active_trees[id]['times_selected'] += 1
            kernel = NUTS(active_trees[id]['T'].model_tree_soft,h_info=info.h_info) # HMC-DFI inner kernel
            bad_chains = True
            num_extra_samps = 0
            while(bad_chains): # keep increasing number of burn in samples until all are good
                mcmc= MCMC(kernel,num_samples=info.num_samps,num_warmup=info.num_warmup+num_extra_samps,num_chains=info.num_chains,progress_bar=False)
                mcmc.run(random.PRNGKey(settings.init_id*1000+ii),data,h=info.h_info['h_init'])
                run_samps_grouped = mcmc.get_samples(group_by_chain=True)
                bad_chains = jnp.any(jnp.var(run_samps_grouped['tau'],axis=1) < 1e-12) or jnp.any(mcmc.last_state.diverging)
                num_extra_samps += 1000
                ii += 1
            # mcmc.print_summary()
            run_samps = mcmc.get_samples()
            active_trees[id]['mcmc'] = mcmc
            active_trees[id]['last_state'] = mcmc.last_state
            active_trees[id]['samples'] = run_samps
            active_trees[id]['log_marg_llh'], active_trees[id]['Ck'], active_trees[id]['tau_tilde'], \
                active_trees[id]['p'], active_trees[id]['dist_logw'] = calculate_marginal_llh(active_trees[id],run_samps,data,settings,info,active=1)

            if(np.isnan(active_trees[id]['log_marg_llh'])): 
                print("log_marg_llh_est is nan -- restarting chains")
                active_trees[id] = restart_chains(active_trees[id],data,settings,info)

        # Select tree with largest utility to sample from
        id = calculate_utility(active_trees,settings,info) # calculate id of tree with largest utility from current set of active trees

        active_trees[id]['times_selected'] += 1

        print("tree selected: ",id)
        print("tree prior: ",active_trees[id]['T'].prior['log_p'])
        print("times proposed: ",active_trees[id]['times_proposed'])
        print("times selected: ",active_trees[id]['times_selected'])
        print("log_marg_llh: ",active_trees[id]['log_marg_llh'])
        
        # Run HMC sampler using last sample as initial value -- perform local inference
        active_trees[id]['mcmc'].post_warmup_state = active_trees[id]['last_state']
        active_trees[id]['mcmc'].run(random.PRNGKey(settings.init_id*1000+ii),data,h=info.h_info['h_final'])
        ii += 1
        # active_trees[id]['mcmc'].print_summary()
        active_trees[id]['last_state'] = active_trees[id]['mcmc'].last_state
        samps = active_trees[id]['mcmc'].get_samples()
        for k in active_trees[id]['samples'].keys():
            active_trees[id]['samples'][k] = np.concatenate((active_trees[id]['samples'][k],samps[k]))

        # Update marginal likelihood based on new samples
        active_trees[id]['log_marg_llh'], active_trees[id]['Ck'], active_trees[id]['tau_tilde'], \
            active_trees[id]['p'], active_trees[id]['dist_logw'] = update_marginal_llh_fast(active_trees[id],samps,data,settings,info,active=1)

        if(np.isnan(active_trees[id]['log_marg_llh'])): 
            print("log_marg_llh_est is nan -- restarting chains")
            active_trees[id] = restart_chains(active_trees[id],data,settings,info)

        print("updated log_marg_llh: ",active_trees[id]['log_marg_llh'])

        # Propose new tree via global proposal from current tree
        u = np.random.rand()
        if(u < (info.move_probs[0]+info.move_probs[1])): # grow or prune move
            tree_new = copy.deepcopy(active_trees[id]['T']) 
            leaves_grow, prob_grow = get_prob_grow(tree_new, info.move_probs[0])#info.prob_grow) 
            if(np.random.rand() < prob_grow): # grow
                node = pyrandom.choice(leaves_grow) # the leaf node we might grow at
                new_left = Node(parent=node,left=None,right=None,ancestors=node.ancestors+[node],path=node.path+[0])
                new_right= Node(parent=node,left=None,right=None,ancestors=node.ancestors+[node],path=node.path+[1])
                node.left = new_left
                node.right = new_right
                new_left.node_id = new_left.get_node_id()	
                new_right.node_id = new_right.get_node_id()
                tree_new.reset_tree(data)
                tree_id = ''.join(str(node.get_node_id()) for node in tree_new.leaf_nodes)
                print("grow move proposed, new id: ",tree_id)
            else: # prune
                nognds = tree_new.get_no_grandchildren()   
                node = pyrandom.choice(nognds)
                node.right = None
                node.left = None
                node.params = {}
                tree_new.reset_tree(data)
                tree_id = ''.join(str(node.get_node_id()) for node in tree_new.leaf_nodes)
                print("prune move proposed, new id: ",tree_id)
            if tree_id in trees.keys():
                trees[tree_id]['times_proposed'] += 1
            else:
                trees[tree_id] = {'T':tree_new,'times_proposed':1,'times_selected':0,'samples':None,'log_marg_llh':None,'log_w':None,'Ck':None,'tau_tilde':None,'dist_logw':None,'p':None}
        else: # global move proposes same tree
            active_trees[id]['times_proposed'] += 1
            print("stay move proposed")

        # Check to see if number of active trees is more than set maximum - remove one with lowest marginal likelihood if so
        if(len(active_trees) > settings.Tmax):
            ids = [id for id in active_trees.keys()]
            min_log_marg = np.argmin([active_trees[id]['log_tau'] for id in active_trees.keys()])
            active_trees.pop(ids[min_log_marg])
        
        # Choose one non-active tree to do local inference
        non_ids = [k for k in trees if k not in active_trees]
        non_id = np.random.choice(non_ids)
        print('\nnon-active tree id: ', non_id)
        trees[non_id]['times_selected'] += 1
        if(trees[non_id]['samples'] is None): # burn-in if needed
            kernel = NUTS(trees[non_id]['T'].model_tree_soft,h_info=info.h_info) # HMC-DFI inner kernel
            bad_chains = True
            num_extra_samps = 0
            while(bad_chains): # keep increasing number of burn in samples until all are good
                mcmc= MCMC(kernel,num_samples=info.num_samps,num_warmup=info.num_warmup+num_extra_samps,num_chains=info.num_chains,progress_bar=False)
                mcmc.run(random.PRNGKey(settings.init_id*1000+ii),data,h=info.h_info['h_init'])
                run_samps_grouped = mcmc.get_samples(group_by_chain=True)
                bad_chains = jnp.any(jnp.var(run_samps_grouped['tau'],axis=1) < 1e-12) or jnp.any(mcmc.last_state.diverging)
                num_extra_samps += 1000
                ii += 1
            # mcmc.print_summary()
            run_samps = mcmc.get_samples()
            trees[non_id]['mcmc'] = mcmc
            trees[non_id]['last_state'] = mcmc.last_state
            trees[non_id]['samples'] = run_samps
            trees[non_id]['log_marg_llh'] = calculate_marginal_llh(trees[non_id],run_samps,data,settings,info,active=0)
        else: # else run local inference
            # Run HMC sampler using last sample as initial value -- perform local inference
            trees[non_id]['mcmc'].post_warmup_state = trees[non_id]['last_state']
            trees[non_id]['mcmc'].run(random.PRNGKey(settings.init_id*1000+ii),data,h=info.h_info['h_final'])
            ii += 1
            # mcmc.print_summary()
            trees[non_id]['last_state'] = trees[non_id]['mcmc'].last_state
            samps = trees[non_id]['mcmc'].get_samples()
            for k in trees[non_id]['samples'].keys():
                trees[non_id]['samples'][k] = np.concatenate((trees[non_id]['samples'][k],samps[k]))
            # Update marginal likelihood based on new samples
            trees[non_id]['log_marg_llh'] = update_marginal_llh_fast(trees[non_id],samps,data,settings,info,active=0)

    time_finish = time.perf_counter()

    if(settings.save == 1):
        # Remove MCMC object - issues with pickling
        for tree in trees:
            try:
                trees[tree].pop('mcmc')
            except: # tree has not yet been considered for local inference
                pass 

        print('file path: ' + settings.out_dir + info.filename)
        results = {'settings': settings, 'mcmc_info': info,
                    'trees':trees,
                    'active_trees':active_trees,
                    'time_total': time_finish-time_start
                }
        pickle.dump(results, open(settings.out_dir+info.filename, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

def get_prob_grow(tree, prob_grow):
    # compute prob of a grow based on number of internal nodes
    leaves = tree.leaf_nodes
    if (len(tree.internal_nodes) == 1): # just one node
        prob = 1
    else:
        prob = prob_grow

    return leaves, prob # leaf nodes, probability of growing

def calculate_utility(trees,settings,info):
    """ Calculates the utility of active trees then returns tree with highest utility score."""
    ids = []
    for k in trees.keys():
        ids.append(k)

    # Find max "exploitation" term across active trees
    max_tau_id = ids[np.argmax([trees[id]['Ck']*trees[id]['tau_tilde'] for id in ids])]
    
    # Calculate sum of all times proposed across all active trees
    sum_Sk = np.sum([trees[id]['times_selected'] for id in ids])

    # Find max "exploration" term across active trees
    if(info.recalculate_p == True):
        print("recalculating p")
        for id in ids:
            trees[id]['p'] = 1 - jnp.power(norm.cdf(info.logw_max,trees[id]['dist_logw']['mu'],trees[id]['dist_logw']['std']),settings.Ta)
        max_p = np.max([trees[id]['p'] for id in ids])
        info.recalculate_p = False
    else:
        max_p = np.max([trees[id]['p'] for id in ids])

    # Iterate over active trees to calculate utility
    u = jnp.zeros(len(ids))
    for ii,k in enumerate(ids):
        Sk = trees[k]['times_selected']
        u = u.at[ii].set((1/Sk) * ((1-settings.delta)*jnp.sqrt(jnp.exp(trees[k]['Ck']-trees[max_tau_id]['Ck']))*(trees[k]['tau_tilde']/trees[max_tau_id]['tau_tilde']) + \
            settings.delta*trees[k]['p']/max_p + settings.beta_opt*np.log(sum_Sk)/np.sqrt(Sk)))  

    if(np.any(np.isnan(u))):
        raise ValueError("nan value in utility function")

    return ids[np.argmax(u)]

def restart_chains(tree_info,data,settings,info):
    # NOTE: This function exists to restart the local sampling routine if chain gets stuck.
    # This can happen in NUTS if the momentum/position is such that it moves to a new mode with the old adaptation hyperparameters.
    # This subsequently causes issues in the marginal likelihood calculation as the variance on the samples is zero.
    global ii
    kernel = NUTS(tree_info['T'].model_tree_soft,h_info=info.h_info)
    bad_chains = True
    num_extra_samps = 0
    while(bad_chains): # keep increasing number of burn in samples until all are good
        mcmc= MCMC(kernel,num_samples=info.num_samps,num_warmup=info.num_warmup+num_extra_samps,num_chains=info.num_chains,progress_bar=False)
        mcmc.run(random.PRNGKey(settings.init_id+100000+ii),data,h=info.h_info['h_init'])
        run_samps_grouped = mcmc.get_samples(group_by_chain=True)
        bad_chains = jnp.any(jnp.var(run_samps_grouped['tau'],axis=1) < 1e-12) or jnp.any(mcmc.last_state.diverging)
        num_extra_samps += 1000
        ii += 1
    # mcmc.print_summary()
    run_samps = mcmc.get_samples()
    tree_info['times_proposed'] = 1 # resets to number of times proposed after first burn-in - allows utility to explore more
    tree_info['mcmc'] = mcmc
    tree_info['last_state'] = mcmc.last_state
    tree_info['samples'] = run_samps
    tree_info['log_marg_llh'], tree_info['Ck'], tree_info['tau_tilde'], \
        tree_info['p'], tree_info['dist_logw'] = calculate_marginal_llh(tree_info,run_samps,data,settings,info,active=1)
    
    if(np.isnan(tree_info['log_marg_llh'])): # this shouldn't ever run, but just in case it's here
        print("log_marg_llh_est is nan -- restarting chains again")
        tree_info = restart_chains(tree_info,data,settings,info)
    
    return tree_info

def calculate_marginal_llh(tree_info,samps,data,settings,info,active):
    """ Calculate initial log_marg_llh_est and corresponding samples """
    global ii # RNG key 
    sbt = StickBreakingTransform()
    st = SigmoidTransform()
    et = ExpTransform()

    if(data['output_type']=='class'):
        sbt_inv_indx = sbt.inv(samps['index'])
        sbt_inv_indx_reshaped = jnp.reshape(sbt_inv_indx,(info.num_chains*info.num_samps,(data['nx']-1)*tree_info['T'].n))
        st_inv_tau = st.inv(samps['tau'])
        params_stacked = jnp.concatenate((st_inv_tau,sbt_inv_indx_reshaped),axis=1)
        num_params = params_stacked.shape[-1]
    else:
        sbt_inv_indx = sbt.inv(samps['index'])
        sbt_inv_indx_reshaped = jnp.reshape(sbt_inv_indx,(info.num_chains*info.num_samps,(data['nx']-1)*tree_info['T'].n))
        st_inv_tau = st.inv(samps['tau'])
        et_inv_sigma = et.inv(samps['sigma'][:,None])
        params_stacked = jnp.concatenate((st_inv_tau,sbt_inv_indx_reshaped,samps['mu'],et_inv_sigma),axis=1)
        num_params = params_stacked.shape[-1]

    # Define covariance matrices across chains for all parameters together - note this is for unconstrained space
    C_all = jnp.zeros((info.num_chains*info.num_samps,num_params,num_params))
    # Compute variance across each parallel chain for each node
    for i in range(info.num_chains):
        C_all = C_all.at[i*info.num_samps:(i+1)*info.num_samps,:,:].set(jnp.repeat(jnp.cov(jnp.transpose(params_stacked[i*info.num_samps:(i+1)*info.num_samps,:]))[None,:,:],info.num_samps,axis=0))

    # 2. Sample from proposal density
    q_all = MultivariateNormal(params_stacked,covariance_matrix=C_all)
    samp_all = q_all.sample(random.PRNGKey(ii),sample_shape=(settings.M,))
    ii += 1
    
    # 3a. Calculate log-prob for each value (denoted pi in paper)
    if(data['output_type']=='class'):
        tau_u = samp_all[:,:,:tree_info['T'].n]
        tau = st(tau_u)
        index_u = jnp.reshape(samp_all[:,:,tree_info['T'].n:],(settings.M,info.num_chains*info.num_samps,tree_info['T'].n,(data['nx']-1)))
        index = sbt(index_u)
        log_prob_eval = log_prob(tree_info['T'].model_tree_soft,{'tau':tau,'index':index},data,settings.h_final,batch_ndims=2)
    else:
        tau_u = samp_all[:,:,:tree_info['T'].n]
        tau = st(tau_u)
        index_u = jnp.reshape(samp_all[:,:,tree_info['T'].n:-tree_info['T'].nl-1],(settings.M,info.num_chains*info.num_samps,tree_info['T'].n,(data['nx']-1)))
        index = sbt(index_u)
        mu = samp_all[:,:,-tree_info['T'].nl-1:-1]
        sigma_u = samp_all[:,:,-1]
        sigma = et(sigma_u)
        log_prob_eval = log_prob(tree_info['T'].model_tree_soft,{'tau':tau,'index':index,'mu':mu,'sigma':sigma},data,settings.h_final,batch_ndims=2)

    logPi_ = jnp.zeros((tau.shape[:-1]))
    for j,val in enumerate(log_prob_eval.values()):
        if((val.ndim == 2)):
            logPi_ += val
        else:
            logPi_ += jnp.sum(val,axis=2)
    logPi = logPi_ + tree_info['T'].prior['log_p']

    # Compute mixture of proposal densities
    # NOTE change of variables information can be found here: https://mc-stan.org/docs/reference-manual/change-of-variables.html
    if(settings.prop_density == 0): # basic - calculation corresponds to Equation 5.13 of (Llorente et al. 2023)
        num_samps = samps['tau'].shape[0] # calculate the current number of samples for this tree
        if(data['output_type']=='class'):
            log_det_tau = np.sum(st.log_abs_det_jacobian(tau_u,tau),axis=2)
            log_det_index = np.sum(sbt.log_abs_det_jacobian(index_u,index),axis=2)
            logPhi = q_all.log_prob(samp_all) - log_det_tau - log_det_index
        else:
            log_det_tau = np.sum(st.log_abs_det_jacobian(tau_u,tau),axis=2)
            log_det_index = np.sum(sbt.log_abs_det_jacobian(index_u,index),axis=2)
            log_det_sigma =  et.log_abs_det_jacobian(sigma_u,sigma)
            logPhi = q_all.log_prob(samp_all) - log_det_tau - log_det_index - log_det_sigma
    else: # spatial (per-chain)  - calculation corresponds to Equation 5.12 of (Llorente et al. 2023)
        num_samps = samps['tau'].shape[0] # calculate the current number of samples for this tree
        logPhi_ = jnp.zeros((settings.M,num_samps,settings.num_samps))
        logPhi = jnp.zeros((settings.M,num_samps))
        params_reshaped = jnp.reshape(samp_all,(settings.M,settings.num_chains,settings.num_samps,num_params))
        def body_fun_class(n,logPhi_):
            samp_all_reshaped = jnp.reshape(jnp.roll(params_reshaped,n,axis=2),(settings.M,settings.num_chains*settings.num_samps,num_params)) 
            tau_reshaped = st(samp_all_reshaped[:,:,:tree_info['T'].n])
            index_reshaped = sbt(samp_all_reshaped[:,:,tree_info['T'].n:])
            tmp = q_all.log_prob(samp_all_reshaped) - np.sum(st.log_abs_det_jacobian(samp_all_reshaped[:,:,:tree_info['T'].n],tau_reshaped),axis=2) \
                - np.sum(sbt.log_abs_det_jacobian(samp_all_reshaped[:,:,tree_info['T'].n:],index_reshaped),axis=2)
            logPhi_ = logPhi_.at[:,:,n].set(tmp)
            return logPhi_
        def body_fun_real(n,logPhi_):
            samp_all_reshaped = jnp.reshape(jnp.roll(params_reshaped,n,axis=2),(settings.M,settings.num_chains*settings.num_samps,num_params)) 
            tau_reshaped = st(samp_all_reshaped[:,:,:tree_info['T'].n])
            index_reshaped = sbt(samp_all_reshaped[:,:,tree_info['T'].n:-tree_info['T'].nl-1])
            sigma_reshaped = et(samp_all_reshaped[:,:,-1])
            tmp = q_all.log_prob(samp_all_reshaped) - np.sum(st.log_abs_det_jacobian(samp_all_reshaped[:,:,:tree_info['T'].n],tau_reshaped),axis=2) \
                - np.sum(sbt.log_abs_det_jacobian(samp_all_reshaped[:,:,tree_info['T'].n:-tree_info['T'].nl-1],index_reshaped),axis=2) - \
                - et.log_abs_det_jacobian(samp_all_reshaped[:,:,-1],sigma_reshaped)
            logPhi_ = logPhi_.at[:,:,n].set(tmp)
            return logPhi_
        if(data['output_type']=='class'):
            logPhi_ = lax.fori_loop(0,settings.num_samps, body_fun_class,logPhi_)
        else:
            logPhi_ = lax.fori_loop(0,settings.num_samps, body_fun_real,logPhi_)
        logPhi = logsumexp(logPhi_,axis=2) - np.log(settings.num_samps)

    # 3. Calculate weights (via Equation 141) and store
    logw = logPi - logPhi

    tree_info['log_w'] = logw
    tree_info['logPi'] = logPi
    tree_info['logPhi'] = logPhi

    if(data['output_type']=='class'):
        tree_info['proposals'] = {}
        tree_info['proposals']['tau'] = tau
        tree_info['proposals']['index'] = index
    else:
        tree_info['proposals'] = {}
        tree_info['proposals']['tau'] = tau
        tree_info['proposals']['index'] = index
        tree_info['proposals']['mu'] = mu
        tree_info['proposals']['sigma'] = sigma

    # Calculate marginal likelihood via these weights (Equation 12 in Appendix)
    log_marg_llh_est = logsumexp(logw) - np.log(settings.M*num_samps) # NOTE num_samps = N*T here

    if(active):
        # Calculate log(mean(log(w)))
        logw_reshaped = jnp.reshape(logw,(settings.M*num_samps))
        n = len(logw_reshaped)
        log_mu = -jnp.log(n) + logsumexp(logw_reshaped)

        # Calculate log(sigma_k)
        tmp = np.vstack([logw_reshaped,np.repeat(log_mu,n)])
        tmp2 = logsumexp(tmp,b=jnp.array([[1],[-1]]),axis=0,return_sign=True)[0]
        log_sigma_k = -jnp.log(n) + logsumexp(2*tmp2)

        # Calculate exploitation term via variance of these weights
        Ck = jnp.max(jnp.array([2*log_marg_llh_est,log_sigma_k]))
        tau_k_tilde = jnp.sqrt(jnp.exp(2*log_marg_llh_est-Ck) + (1+settings.kappa)*jnp.exp(log_sigma_k-Ck))

        # Calculate exploration term
        # First check if any new logw value is larger than then the current global max(logw)
        logw_max = jnp.max(logw)
        if(logw_max > info.logw_max):
            info.logw_max = logw_max
            info.recalculate_p = True

        # Calculate hyperparameters of logw density - assuming normal distribution
        logw_mu = jnp.mean(logw)
        logw_std = jnp.std(logw)
        dist_logw = {'mu':logw_mu,'std':logw_std}
        p_k = 1 - jnp.power(norm.cdf(info.logw_max,logw_mu,logw_std),settings.Ta)

        return log_marg_llh_est, Ck, tau_k_tilde, p_k, dist_logw
    else:
        return log_marg_llh_est

def update_marginal_llh_fast(tree_info,samps,data,settings,info,active):
    global ii # RNG key 
    sbt = StickBreakingTransform()
    st = SigmoidTransform()
    et = ExpTransform()

    if(data['output_type']=='class'):
        sbt_inv_indx = sbt.inv(samps['index'])
        sbt_inv_indx_reshaped = jnp.reshape(sbt_inv_indx,(info.num_chains*info.num_samps,(data['nx']-1)*tree_info['T'].n))
        st_inv_tau = st.inv(samps['tau'])
        params_stacked = jnp.concatenate((st_inv_tau,sbt_inv_indx_reshaped),axis=1)
        num_params = params_stacked.shape[-1]
    else:
        sbt_inv_indx = sbt.inv(samps['index'])
        sbt_inv_indx_reshaped = jnp.reshape(sbt_inv_indx,(info.num_chains*info.num_samps,(data['nx']-1)*tree_info['T'].n))
        st_inv_tau = st.inv(samps['tau'])
        et_inv_sigma = et.inv(samps['sigma'][:,None])
        params_stacked = jnp.concatenate((st_inv_tau,sbt_inv_indx_reshaped,samps['mu'],et_inv_sigma),axis=1)
        num_params = params_stacked.shape[-1]

    # Define covariance matrices across chains for all parameters together - note this is for unconstrained space
    C_all = jnp.zeros((info.num_chains*info.num_samps,num_params,num_params))

    # Compute variance across each parallel chain for each node
    for i in range(info.num_chains):
        C_all = C_all.at[i*info.num_samps:(i+1)*info.num_samps,:,:].set(jnp.repeat(jnp.cov(jnp.transpose(params_stacked[i*info.num_samps:(i+1)*info.num_samps,:]))[None,:,:],info.num_samps,axis=0))

    # 2. Sample from proposal density
    q_all = MultivariateNormal(params_stacked,covariance_matrix=C_all)
    samp_all = q_all.sample(random.PRNGKey(ii),sample_shape=(settings.M,))
    ii += 1

    # 3a. Calculate log-prob for each value (denoted pi in paper)
    if(data['output_type']=='class'):
        tau_u = samp_all[:,:,:tree_info['T'].n]
        tau = st(tau_u)
        index_u = jnp.reshape(samp_all[:,:,tree_info['T'].n:],(settings.M,info.num_chains*info.num_samps,tree_info['T'].n,(data['nx']-1)))
        index = sbt(index_u)
        log_prob_eval = log_prob(tree_info['T'].model_tree_soft,{'tau':tau,'index':index},data,settings.h_final,batch_ndims=2)
    else:
        tau_u = samp_all[:,:,:tree_info['T'].n]
        tau = st(tau_u)
        index_u = jnp.reshape(samp_all[:,:,tree_info['T'].n:-tree_info['T'].nl-1],(settings.M,info.num_chains*info.num_samps,tree_info['T'].n,(data['nx']-1)))
        index = sbt(index_u)
        mu = samp_all[:,:,-tree_info['T'].nl-1:-1]
        sigma_u = samp_all[:,:,-1]
        sigma = et(sigma_u)
        log_prob_eval = log_prob(tree_info['T'].model_tree_soft,{'tau':tau,'index':index,'mu':mu,'sigma':sigma},data,settings.h_final,batch_ndims=2)

    logPi_ = jnp.zeros((tau.shape[:-1]))
    for j,val in enumerate(log_prob_eval.values()):
        if((val.ndim == 2)):
            logPi_ += val
        else:
            logPi_ += jnp.sum(val,axis=2)
    logPi = logPi_ + tree_info['T'].prior['log_p']

    # Compute mixture of proposal densities
    # NOTE change of variables information can be found here: https://mc-stan.org/docs/reference-manual/change-of-variables.html
    if(settings.prop_density == 0): # basic - calculation corresponds to Equation 5.13 of (Llorente et al. 2023)
        num_samps = samps['tau'].shape[0] # calculate the current number of samples for this tree
        if(data['output_type']=='class'):
            log_det_tau = np.sum(st.log_abs_det_jacobian(tau_u,tau),axis=2)
            log_det_index = np.sum(sbt.log_abs_det_jacobian(index_u,index),axis=2)
            logPhi = q_all.log_prob(samp_all) - log_det_tau - log_det_index
        else:
            log_det_tau = np.sum(st.log_abs_det_jacobian(tau_u,tau),axis=2)
            log_det_index = np.sum(sbt.log_abs_det_jacobian(index_u,index),axis=2)
            log_det_sigma =  et.log_abs_det_jacobian(sigma_u,sigma)
            logPhi = q_all.log_prob(samp_all) - log_det_tau - log_det_index - log_det_sigma
    else: # spatial (per-chain)  - calculation corresponds to Equation 5.12 of (Llorente et al. 2023)
        num_samps = samps['tau'].shape[0] # calculate the current number of samples for this tree
        logPhi_ = jnp.zeros((settings.M,num_samps,settings.num_samps))
        logPhi = jnp.zeros((settings.M,num_samps))
        params_reshaped = jnp.reshape(samp_all,(settings.M,settings.num_chains,settings.num_samps,num_params))
        def body_fun_class(n,logPhi_):
            samp_all_reshaped = jnp.reshape(jnp.roll(params_reshaped,n,axis=2),(settings.M,settings.num_chains*settings.num_samps,num_params)) 
            tau_reshaped = st(samp_all_reshaped[:,:,:tree_info['T'].n])
            index_reshaped = sbt(samp_all_reshaped[:,:,tree_info['T'].n:])
            tmp = q_all.log_prob(samp_all_reshaped) - np.sum(st.log_abs_det_jacobian(samp_all_reshaped[:,:,:tree_info['T'].n],tau_reshaped),axis=2) \
                - np.sum(sbt.log_abs_det_jacobian(samp_all_reshaped[:,:,tree_info['T'].n:],index_reshaped),axis=2)
            logPhi_ = logPhi_.at[:,:,n].set(tmp)
            return logPhi_
        def body_fun_real(n,logPhi_):
            samp_all_reshaped = jnp.reshape(jnp.roll(params_reshaped,n,axis=2),(settings.M,settings.num_chains*settings.num_samps,num_params)) 
            tau_reshaped = st(samp_all_reshaped[:,:,:tree_info['T'].n])
            index_reshaped = sbt(samp_all_reshaped[:,:,tree_info['T'].n:-tree_info['T'].nl-1])
            sigma_reshaped = et(samp_all_reshaped[:,:,-1])
            tmp = q_all.log_prob(samp_all_reshaped) - np.sum(st.log_abs_det_jacobian(samp_all_reshaped[:,:,:tree_info['T'].n],tau_reshaped),axis=2) \
                - np.sum(sbt.log_abs_det_jacobian(samp_all_reshaped[:,:,tree_info['T'].n:-tree_info['T'].nl-1],index_reshaped),axis=2) - \
                - et.log_abs_det_jacobian(samp_all_reshaped[:,:,-1],sigma_reshaped)
            logPhi_ = logPhi_.at[:,:,n].set(tmp)
            return logPhi_
        if(data['output_type']=='class'):
            logPhi_ = lax.fori_loop(0,settings.num_samps, body_fun_class,logPhi_)
        else:
            logPhi_ = lax.fori_loop(0,settings.num_samps, body_fun_real,logPhi_)
        logPhi = logsumexp(logPhi_,axis=2) - np.log(settings.num_samps)

    # 3. Calculate weights (via Equation 141) and store
    logw = logPi - logPhi

    tree_info['log_w'] = jnp.hstack((tree_info['log_w'],logw))

    if(data['output_type']=='class'):
        tree_info['proposals']['tau'] = np.concatenate((tree_info['proposals']['tau'],tau),axis=1)
        tree_info['proposals']['index'] = np.concatenate((tree_info['proposals']['index'],index),axis=1)
    else:
        tree_info['proposals']['tau'] = np.concatenate((tree_info['proposals']['tau'],tau),axis=1)
        tree_info['proposals']['index'] = np.concatenate((tree_info['proposals']['index'],index),axis=1)
        tree_info['proposals']['mu'] = np.concatenate((tree_info['proposals']['mu'],mu),axis=1)
        tree_info['proposals']['sigma'] = np.concatenate((tree_info['proposals']['sigma'],sigma),axis=1)

    # Calculate marginal likelihood via these weights (Equation 12 in Appendix)
    total_samps = tree_info['samples']['tau'].shape[0]
    logw_reshaped = jnp.reshape(logw,(settings.M*num_samps))
    log_marg_llh_est = logsumexp(jnp.hstack([tree_info['log_marg_llh']+jnp.log(settings.M*(total_samps-num_samps)),logw_reshaped]))-jnp.log(settings.M*total_samps)

    if(active):
        # Calculate log(mean(log(w)))
        logw_all_reshaped = jnp.reshape(tree_info['log_w'],(settings.M*total_samps))
        n = len(logw_all_reshaped)
        log_mu = -jnp.log(n) + logsumexp(logw_all_reshaped)

        # Calculate log(sigma_k)
        tmp = np.vstack([logw_all_reshaped,np.repeat(log_mu,n)])
        tmp2 = logsumexp(tmp,b=jnp.array([[1],[-1]]),axis=0,return_sign=True)[0]
        log_sigma_k = -jnp.log(n) + logsumexp(2*tmp2)

        # Calculate exploitation term via variance of these weights
        Ck = jnp.max(jnp.array([2*log_marg_llh_est,log_sigma_k]))
        tau_k_tilde = jnp.sqrt(jnp.exp(2*log_marg_llh_est-Ck) + (1+settings.kappa)*jnp.exp(log_sigma_k-Ck))

        # Calculate exploration term
        # First check if any new logw value is larger than then the current global max(logw)
        logw_max = jnp.max(logw)
        if(logw_max > info.logw_max):
            info.logw_max = logw_max
            info.recalculate_p = True

        # Calculate hyperparameters of logw density - assuming normal distribution
        logw_mu = jnp.mean(logw_all_reshaped)
        logw_std = jnp.std(logw_all_reshaped)
        dist_logw = {'mu':logw_mu,'std':logw_std}
        p_k = 1 - jnp.power(norm.cdf(info.logw_max,logw_mu,logw_std),settings.Ta)

        return log_marg_llh_est, Ck, tau_k_tilde, p_k, dist_logw
    else:
        return log_marg_llh_est

if __name__ == "__main__":
    main()