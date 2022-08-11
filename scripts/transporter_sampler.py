import numpy as np
import tellurium as te
import multiprocessing as mp
import emcee 
import matplotlib.pyplot as plt
import time
import pickle
from datetime import datetime
import os
import json

mp.set_start_method('fork')


def calc_norm_log_like(mu,sigma,X):
    ''' calculates the Normal log-likelihood function: -[(n/2)ln(2pi*sigma^2)]-[sum((X-mu)^2)/(2*sigma^2)]
    ref: https://www.statlect.com/fundamentals-of-statistics/normal-distribution-maximum-likelihood 
    '''
    # fix this - remove loop
    n = len(X)
    f1 = -1*(n/2)*np.log(2*np.pi*sigma**2)
    f2_a = -1/(2*sigma**2)
    f2_b = 0 
    for i in range(n):
        f2_b += (X[i]-mu[i])**2
    f2 = f2_a*f2_b
    log_likelihood = f1+f2
    return log_likelihood


def calc_log_like(K,y_obs,m):
    '''calculates the log likelihood of a transporter tellurium ODE model m, given data y_obs, and parameters K
    '''
   
    idx_list = [0,2,4,6,8]  # index of rate pairs used to set attribute, last rate omitted - fix this later 
    m.resetToOrigin()
    m.H_out = 5e-7
    m.integrator.absolute_tolerance = 1e-18
    m.integrator.relative_tolerance = 1e-12

    # update tellurium model parameter values (rate constants)
    for i, idx in enumerate(idx_list):
        setattr(m, f'k{i+1}_f', 10**K[idx])
        setattr(m, f'k{i+1}_r', 10**K[idx+1])

    # last rate constant (k6_r) has cycle constraint
    m.k6_f = 10**K[10]
    m.k6_r = (m.k1_f*m.k2_f*m.k3_f*m.k4_f*m.k5_f*m.k6_f)/(m.k1_r*m.k2_r*m.k3_r*m.k4_r*m.k5_r)

    try:
        D_tmp = m.simulate(0, 5, 125, selections=['time', 'rxn4'])
        y_tmp = D_tmp['rxn4'][1:]  # remove first point
        sigma = 10**K[11]
        log_like_tmp = calc_norm_log_like(y_tmp,sigma,y_obs)
    except:
        log_like_tmp = -np.inf
    return log_like_tmp


def calc_log_prior(p):
    '''calculates the log of uniform prior distribution
    if parameter is in range, the probability is log(1)-->0, else log(0)-->-inf
    '''

    # fix this later - should connect w/ parameter info data file
    b = 2  # shift parameter priors by 2 (orders of magnitude) to be non-centered
    lb = np.array([6, -1, -2, -2, 3, -1, -1, 6, -2, -2, -1]) + b  # log10 rate constant prior lower bound + shift
    ub = np.array([12, 5, 4, 4, 9, 5, 5, 12, 4, 4, 5]) + b  # log10 rate constant prior upper bound + shift
    sigma_lb = np.log10(5e-14)  # log10 noise sigma prior lower bound 
    sigma_ub = np.log10(5e-13)  # log10 noise sigma prior upper bound 
    if ((lb < p[:-1]) & (p[:-1] < ub)).all() and (sigma_lb<p[-1]<sigma_ub):
        return 0
    else:
        return -np.inf


def calc_log_post(theta, y_obs, extra_parameters):
    '''calculate the log of the posterior probability
    log posterior = log likelihood*beta + log prior^beta
    where beta is used for tempering
    '''
    m = extra_parameters[0]
    beta = extra_parameters[1]
    log_pr = calc_log_prior(theta)
    if not np.isfinite(log_pr):
        return -np.inf  # ~zero probability
    log_like = calc_log_like(theta, y_obs, m)
    if not np.isfinite(log_like):
        return -np.inf  # ~zero probability 
    else:
        log_prob = log_pr + beta*log_like  
        return log_prob


def get_p0(b_list, n):
    '''get initial uniform distributed samples using boundaries from b_list, and number of samples n
    b_list[i][0] = parameter lower bound, b_list[i][1] = parameter upper bound
    '''
    p0_array = np.transpose(np.array([np.random.uniform(b[0],b[1],n) for b in b_list]))  # re-arrange array for sampler
    return p0_array


def parse_p_info(p_info, near_global_min=True):
    '''parse parameter settings data
    p_info[i] = [parameter name, lower bound, upper bound, reference value]
    '''
    p_ref = [p_i[3] for p_i in p_info]
    p_labels = [p[0] for p in p_info]
    if near_global_min==True:
        p_bounds = [(p[3]*0.999, p[3]*1.001) if p[3] > 0 else (p[3]*1.001, p[3]*0.999) for p in p_info]  # near global min
    else:
        p_bounds = [(p[1], p[2]) for p in p_info]  # default
    return p_ref, p_labels, p_bounds


def wrapper(arg_list):
    '''wrapper function to make and run ensemble sampler object in parallel'''


    log_prob = arg_list[0]
    log_prob_args = arg_list[1]
    p_0 = arg_list[2]
    antimony_string_SS = arg_list[3]
    backend_fname = arg_list[4][0]
    backend_rname = arg_list[4][1]  
    n_dim = int(arg_list[5][0])
    n_walkers = int(arg_list[5][1])
    n_steps = int(arg_list[5][2])
    new_roadrunner = te.loada(antimony_string_SS)
    log_prob_args[-1][0] = new_roadrunner
    backend = emcee.backends.HDFBackend(backend_fname, name=backend_rname)
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob, args=log_prob_args, backend=backend)   
    # moves=[
    #     (emcee.moves.DEMove(), 0.2),
    #     (emcee.moves.DESnookerMove(), 0.2),
    #     (emcee.moves.KDEMove(), 0.2),
    #     (emcee.moves.WalkMove(), 0.2),
    #     (emcee.moves.StretchMove(), 0.2),
    # ])
    state = sampler.run_mcmc(p_0,n_steps)
    return (sampler, state)



if __name__ == "__main__":

    ### input arguments
    model_file = "/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/transporter_model/antiporter_12D_model.txt"
    obs_data_file = "/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/synthetic_data/synth_data_1exp_a_trunc.csv"
    parameter_file = "/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/transporter_model/12D_transporter_w_full_priors.json"
    parallel = False
    seed = 42
    n_parallel = 6
    n_walkers = 100
    n_dim = 12
    n_steps = 10
    n_shuffles = 10
    thin = 1
    n_ensembles = n_parallel
    np.random.seed(seed)

    ### file i/o - create new directory, load tellurium model string, and load model parameter info
    date_string = datetime.today().strftime('%Y%m%d_%H%M%S')
    out_fname=f'run_d{date_string}_p{parallel}_np{n_parallel}_nw{n_walkers}_ns{n_steps}_nd{n_dim}_nsh{n_shuffles}_t{thin}_r{seed}'
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, out_fname)
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    with open(model_file, "r") as f:
        antimony_string_SS = f.read()
    with open (parameter_file, 'rb') as fp:
        p_info = json.load(fp)
    p_ref, p_labels, p_bounds = parse_p_info(p_info, near_global_min=False)
    _, _, p_bounds2 = parse_p_info(p_info, near_global_min=False)  # for plot (useful if starting near global max)

    ### set log likelihood arguments and initial parameter sets
    y_obs_list = [np.genfromtxt(obs_data_file) for i in range(n_parallel)]
    log_post_args = [[y_obs_list[i], [None, 1]] for i in range(n_parallel)]  # replace 'none' later w/ tellurium model (roadrunner)
    backend_fname_list = [f"{final_directory}/ensemble_{i}.h5" for i in range(n_parallel)]
    p_0 = [get_p0(p_bounds, n_walkers) for i in range(n_parallel)]

    ### write to log file
    with open(os.path.join(final_directory, f'{out_fname}_log.txt'), "a") as f:
        f.write(f"date: {date_string}\n")
        f.write(f"model file: {model_file}\n")
        f.write(f"parameter file: {parameter_file}\n")
        f.write(f"data file: {obs_data_file}\n")
        f.write(f"parallel: {parallel}\n")
        f.write(f"seed: {seed}\n")
        f.write(f"n parallel: {n_parallel}\n")
        f.write(f"n walkers: {n_walkers}\n")
        f.write(f"n dim: {n_dim}\n")
        f.write(f"n steps: {n_steps}\n")
        f.write(f"n shuffles: {n_shuffles}\n")
        f.write(f"thin by: {thin}\n")
        f.write(f"out fname: {out_fname}\n")
        f.write(f"parameter ref: {p_ref}\n")
        f.write(f"parameter labels: {p_labels}\n")
        f.write(f"parameter boundaries: {p_bounds}\n")
        f.write(f"initial parameters p_0[0]: {p_0[0]}\n")

    ### serial affine invariant ensemble sampler (using Emcee)
    if parallel == False:
        # run sampler
        backend_label = (backend_fname_list[0],"shuffle_0")
        arg_list = (calc_log_post, log_post_args[0], p_0[0], antimony_string_SS, (backend_fname_list[0],f"shuffle_0.h5"),(n_dim, n_walkers, n_steps))
        t0 = time.time()
        serial_result = wrapper(arg_list)   
        wallclock = time.time()-t0
        serial_samples = serial_result[0].get_chain(flat=True, thin=thin)

        # plot histogram and save data
        n_bins = 100
        fig, axs = plt.subplots(4,3, figsize=(15,15))
        for i, ax in enumerate(axs.flatten()):  # for each subplot figure (parameter)  
            D_tmp = np.transpose(serial_samples)[i] 
            ax.hist(D_tmp, n_bins, histtype="step", density=True, alpha=0.85, color='k')   # plot parameter histogram
            ax.set_title(f'p_{i} distribution')
            ax.set_xlim(p_bounds2[i][0], p_bounds2[i][1])
            ax.axvline(p_ref[i], 0,1, ls='--', color='k')
        plt.suptitle('1D parameter distributions - serial run')
        plt.tight_layout()
        plt.savefig(f'{final_directory}/{out_fname}_distributions.png')
        plt.close()
        np.savetxt(f"{final_directory}/{out_fname}_samples.csv", serial_samples, delimiter=",")
    
    ### parallel affine invariant ensemble sampler (using Emcee) w/ shuffling
    else:
        # run sampler 
        t0 = time.time()
        sample_list = []
        for i in range(n_shuffles):
            parallel_arg_list = [
                (calc_log_post, log_post_args[j], p_0[j], antimony_string_SS, (backend_fname_list[j],f"shuffle_{i}.h5"), (n_dim, n_walkers, n_steps)) for j in range(n_parallel)]
            with mp.Pool(n_parallel) as pool:
                parallel_result = pool.map(wrapper, parallel_arg_list)
            samples_tmp = [_result[0].get_chain(flat=True, thin=thin) for _result in parallel_result]
            last_samples = [_result[0].get_chain()[-1] for _result in parallel_result]
            agg_samples = np.reshape(last_samples, (n_ensembles*n_walkers, n_dim))
            np.random.shuffle(agg_samples)
            p_0 = np.reshape(agg_samples, (n_ensembles, n_walkers, n_dim))
            sample_list.append(samples_tmp)
        wallclock = time.time()-t0
        
        # plot histogram and save data
        with open(f'{final_directory}/{out_fname}_sample_list.pickle', 'wb') as fp:
            pickle.dump(sample_list, fp)
        n_bins = 100
        cmap = plt.get_cmap('inferno')
        color = [cmap(1.*i/n_shuffles) for i in range(n_shuffles)]
        fig, axs = plt.subplots(4,3, figsize=(15,15))
        for k in range(n_shuffles):
            tmp = sample_list[k]
            for i, ax in enumerate(axs.flatten()):  # for each subplot figure (parameter)
                s = np.array(0.0)
                n = np.array(0.0)
                for j in range(n_ensembles):
                    D_tmp = np.transpose(tmp[j])[i]
                    s = s + D_tmp
                avg = s/n_ensembles 
                ax.hist(avg, n_bins, histtype="step", density=True, alpha=0.85, color=color[k])   # plot parameter histogram
                ax.set_title(f'p_{i} distribution')
                ax.set_xlim(p_bounds2[i][0], p_bounds2[i][1])
                ax.axvline(p_ref[i], 0,1, ls='--', color='k')
        plt.suptitle('Average ensemble 1D parameter distributions')
        plt.tight_layout()
        plt.savefig(f'{final_directory}/{out_fname}_distributions.png')
        plt.close()

    ### write wall clock time to file
    with open(os.path.join(final_directory, f'{out_fname}_log.txt'), "a") as f:
        f.write(f"wall clock: {wallclock} sec\n")