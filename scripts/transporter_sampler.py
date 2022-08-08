import numpy as np
import tellurium as te
import multiprocessing as mp
import emcee 
import matplotlib.pyplot as plt
import time
import pickle

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

    n_walkers = 2000 # for test
    n_dim = 12
    #n_steps = 10
    n_steps=int(10)  # serial mode n_steps = n_steps * n_mixing_stages
    log_prob = arg_list[0]
    log_prob_args = arg_list[1]
    p_0 = arg_list[2]
    antimony_string_SS = arg_list[3]
    new_roadrunner = te.loada(antimony_string_SS)
    log_prob_args[-1][0] = new_roadrunner
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob, args=log_prob_args,)   
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


    # seed = 42
    # np.random.seed(seed)
    # ### model settings
    # n_parallel = 4
    # model_file = "antiporter_12D_model.txt"
    # obs_data_file = "data_grid_test3_1exp_v2.csv"

    # with open(model_file, "r") as f:
    #     antimony_string_SS = f.read()
    # y_obs_list = [np.genfromtxt(obs_data_file) for i in range(n_parallel)]
    # log_post_args = [[y_obs_list[i], [None, 1]] for i in range(n_parallel)]  # replace 'none' later w/ tellurium model (roadrunner)

    # # fix this - make seperate file
    # p_info = [   
    #     ["log_k1_f",6,12,10],
    #     ["log_k1_r",-1,5,3],
    #     ["log_k2_f",-2,4,2],
    #     ["log_k2_r",-2,4,2],
    #     ["log_k3_f",3,9,7],
    #     ["log_k3_r",-1,5,3],
    #     ["log_k4_f",-1,5,3],
    #     ["log_k4_r",6,12,10],
    #     ["log_k5_f",-2,4,2],
    #     ["log_k5_r",-2,4,2],
    #     ["log_k6_f",-1,5,3],
    #     ["log_sigma",np.log10(5e-14), np.log10(5e-13), -13],
    # ]  
    # p_ref, p_labels, p_bounds = parse_p_info(p_info, near_global_min=True)
    # p_0 = [get_p0(p_bounds, 100) for i in range(n_parallel)]
    # print(np.shape(p_0))

    # arg_list = (calc_log_post, log_post_args[0], p_0[0], antimony_string_SS)

    # # Serial sampling
    # serial_result = wrapper(arg_list)
    # serial_log_probs = serial_result[0].get_log_prob(flat=True)
    # assert serial_log_probs[0] > 0

    # # Parallel sampling
    # parallel_arg_list = [(calc_log_post, log_post_args[i], p_0[i], antimony_string_SS) for i in range(n_parallel)]
    # with mp.Pool(n_parallel) as pool:
    #     parallel_result = pool.map(wrapper, parallel_arg_list)
    # parallel_log_probs = [_result[0].get_log_prob(flat=True) for _result in parallel_result]
    # for i in range(n_parallel):
    #     plt.plot(parallel_log_probs[i], alpha=0.75, label=f'parallel ensemble {i}')
    # plt.plot(serial_log_probs, color='k', label=f'serial ensemble')
    # plt.title('Aggregated enesmble log probabilities vs iteration')
    # plt.legend()
    # plt.savefig('test.png')
    # plt.clf()
    # for i in range(n_parallel):
    #     plt.hist(parallel_log_probs[i], bins=100, density=True, label=f'parallel ensemble {i}', histtype='step', alpha=0.75)
    # plt.hist(serial_log_probs, bins=100, color='k', density=True, label=f'serial ensemble', histtype='step', alpha=0.75)
    # plt.title('ensemble log probability distributions')
    # plt.legend()
    # plt.savefig('test2.png')


    out_fname='test_parallel_long_2'
    seed = 42
    np.random.seed(seed)

    n_parallel = 6
    model_file = "antiporter_12D_model.txt"
    obs_data_file = "data_grid_test3_1exp_v2.csv"
    n_ensembles = n_parallel
    n_walkers = 2000
    n_dim = 12
    n_steps = 10
    n_shuffles = 1000

    with open(model_file, "r") as f:
        antimony_string_SS = f.read()
    y_obs_list = [np.genfromtxt(obs_data_file) for i in range(n_parallel)]
    log_post_args = [[y_obs_list[i], [None, 1]] for i in range(n_parallel)]  # replace 'none' later w/ tellurium model (roadrunner)

    # fix this - make seperate file
    p_info = [   
        ["log_k1_f",6,12,10],
        ["log_k1_r",-1,5,3],
        ["log_k2_f",-2,4,2],
        ["log_k2_r",-2,4,2],
        ["log_k3_f",3,9,7],
        ["log_k3_r",-1,5,3],
        ["log_k4_f",-1,5,3],
        ["log_k4_r",6,12,10],
        ["log_k5_f",-2,4,2],
        ["log_k5_r",-2,4,2],
        ["log_k6_f",-1,5,3],
        ["log_sigma",np.log10(5e-14), np.log10(5e-13), -13],
    ]  
    p_ref, p_labels, p_bounds = parse_p_info(p_info, near_global_min=False)
    _, _, p_bounds2 = parse_p_info(p_info, near_global_min=False)
    p_0 = [get_p0(p_bounds, n_walkers) for i in range(n_parallel)]
    #print(np.shape(p_0))

    # # Serial sampling
    # arg_list = (calc_log_post, log_post_args[0], p_0[0], antimony_string_SS)
    # t0 = time.time()
    # serial_result = wrapper(arg_list)
    # print(f'{time.time()-t0} sec')
    # # serial_log_probs = serial_result[0].get_log_prob(flat=True)
    # # #assert serial_log_probs[0] > 0
    # serial_samples = serial_result[0].get_chain(flat=True)
    
    # n_bins = 100
    # fig, axs = plt.subplots(4,3, figsize=(15,15))
    # for i, ax in enumerate(axs.flatten()):  # for each subplot figure (parameter)
        
    #     D_tmp = np.transpose(serial_samples)[i] 
    #     ax.hist(D_tmp, n_bins, histtype="step", density=True, alpha=0.85, color='k')   # plot parameter histogram
    #     ax.set_title(f'p_{i} distribution')
    #     ax.set_xlim(p_bounds2[i][0], p_bounds2[i][1])
    #     ax.axvline(p_ref[i], 0,1, ls='--', color='k')
    # plt.suptitle('1D parameter distributions - serial run')
    # plt.tight_layout()
    # plt.savefig(f'test_serial.png')
    # plt.close()
    # np.savetxt("test_serial.csv", serial_samples, delimiter=",")



    # Parallel sampling
    t0 = time.time()
    sample_list = []
    log_prob_list = []
    for i in range(n_shuffles):
        parallel_arg_list = [(calc_log_post, log_post_args[i], p_0[i], antimony_string_SS) for i in range(n_parallel)]
        with mp.Pool(n_parallel) as pool:
            parallel_result = pool.map(wrapper, parallel_arg_list)
        samples_tmp = [_result[0].get_chain(flat=True) for _result in parallel_result]
        log_prob_tmp = [_result[0].get_log_prob(flat=True) for _result in parallel_result]
        last_samples = [_result[0].get_chain()[-1] for _result in parallel_result]
        agg_samples = np.reshape(last_samples, (n_ensembles*n_walkers, n_dim))
        np.random.shuffle(agg_samples)
        p_0 = np.reshape(agg_samples, (n_ensembles, n_walkers, n_dim))
        sample_list.append(samples_tmp)
        log_prob_list.append(log_prob_tmp)
    print(f'{time.time()-t0} sec')
    print(np.shape(sample_list))

    
    with open(f'{out_fname}.pickle', 'wb') as fp:
        pickle.dump(sample_list, fp)
    
    #with open (f'{out_fname}.pickle', 'rb') as fp:
    #    sample_list = pickle.load(fp)

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
                #ax.hist(D_tmp, n_bins, histtype="step", density=True, label=f'idx {j}', alpha=0.75)   # plot parameter histogram
            avg = s/n_ensembles 
            ax.hist(avg, n_bins, histtype="step", density=True, alpha=0.85, color=color[k])   # plot parameter histogram
            
            #ax.hist(s/n, n_bins, histtype="step", density=True, label=f'avg', color='k')
            #ax.legend()
            ax.set_title(f'p_{i} distribution')
            ax.set_xlim(p_bounds2[i][0], p_bounds2[i][1])
            ax.axvline(p_ref[i], 0,1, ls='--', color='k')
    plt.suptitle('Average ensemble 1D parameter distributions')
    plt.tight_layout()
    plt.savefig(f'{out_fname}.png')
    plt.close()