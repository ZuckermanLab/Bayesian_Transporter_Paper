import numpy as np
import tellurium as te
import matplotlib.pyplot as plt
import time
from datetime import datetime
import os
import json
import argparse
import multiprocessing as mp
import sys
import pocomc as pc


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
    
    t0 = 0
    tf = 3
    n_technical_replicas = 1
    H_out_list = [5e-7]  #  experiments w/ varying external pH conditions [5e-7,0.2e-7]

    k_dict = {
        "1":[0,1],
        "2":[2,3],
        "3":[4,5],
        "4":[6,7],
        "5":[8,9],
        "8":[12,13]
    }
    y_list = []  # empty list to store flux from different experiments
    sigma = 10**K[-1]  # noise standard deviation is last parameter term

    for i in range(len(H_out_list)):
        m.resetToOrigin()
        m.H_act = H_out_list[i]  # plot values
        m.integrator.absolute_tolerance = 1e-22
        m.integrator.relative_tolerance = 1e-12

        # update tellurium model parameter values (rate constants)
        for key, value in k_dict.items():
            setattr(m, f'k{key}_f', 10**K[value[0]])
            setattr(m, f'k{key}_r', 10**K[value[1]])

        # k6 and k7 have cycle constraints
        m.k6_f = 10**K[10]
        m.k7_f = 10**K[11]

        try:
            D_tmp = m.simulate(t0, tf, 180, selections=['time', 'current'])
            y_tmp = D_tmp['current'][61:]  # remove first stage
        except:
            log_like_tmp = -np.inf  # if there is an issue calculating the flux --> no probability
            return log_like_tmp

        y_list.append(y_tmp)
    if n_technical_replicas > 1:
        y_sim = np.hstack(y_list*n_technical_replicas)
    else:
        y_sim = np.hstack(y_list)
    log_like_tmp = calc_norm_log_like(y_sim,sigma,y_obs)
    return log_like_tmp


def calc_log_prior(p):
    '''calculates the log of uniform prior distribution
    if parameter is in range, the probability is log(1)-->0, else log(0)-->-inf
    '''

    # fix this later - should connect w/ parameter info data file
    b = 0  # shift parameter priors by 2 (orders of magnitude) to be non-centered
    lb = np.array([6, -1, -2, -2, 3, -1, -1, 6, -2, -2, -1, 6, -1, 3]) + b  # log10 rate constant prior lower bound + shift
    ub = np.array([12, 5, 4, 4, 9, 5, 5, 12, 4, 4, 5, 12, 5, 9]) + b  # log10 rate constant prior upper bound + shift
    sigma_lb = np.log10(5e-12) # log10 noise sigma prior lower bound 
    sigma_ub = np.log10(5e-11)  # log10 noise sigma prior upper bound 
    if ((lb < p[:-1]) & (p[:-1] < ub)).all() and (sigma_lb<p[-1]<sigma_ub):
        return 0
    else:
        return -np.inf


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
    return p_ref, p_labels, np.array(p_bounds)


def synthesize_data(K,m):
    '''calculates the log likelihood of a transporter tellurium ODE model m, given data y_obs, and parameters K
    '''

    t0 = 0
    tf = 3
    n_technical_replicas = 1
    H_out_list = [5e-7]  #  experiments w/ varying external pH conditions [5e-7,0.2e-7]
    k_dict = {
        "1":[0,1],
        "2":[2,3],
        "3":[4,5],
        "4":[6,7],
        "5":[8,9],
        "8":[12,13]
    }
    y_list = []  # empty list to store flux from different experiments
    sigma = 10**K[-1]  # noise standard deviation is last term

    for i in range(len(H_out_list)):
        m.resetToOrigin()
        m.H_act = H_out_list[i]  # plot values
        m.integrator.absolute_tolerance = 1e-22
        m.integrator.relative_tolerance = 1e-12

        # update tellurium model parameter values (rate constants)
        for key, value in k_dict.items():
            setattr(m, f'k{key}_f', 10**K[value[0]])
            setattr(m, f'k{key}_r', 10**K[value[1]])

        # k6 and k7 have cycle constraints
        m.k6_f = 10**K[10]
        m.k7_f = 10**K[11]
   
        try:
            D_tmp = m.simulate(t0, tf, 180, selections=['time', 'current'])
            y_tmp = D_tmp['current'][61:]  # remove first stage
        except:
            print('error simulating model')
            sys.exit(1) 
        y_list.append(y_tmp)

    if n_technical_replicas > 1:
        y_true = np.hstack(y_list*n_technical_replicas)
    else:
        y_true = np.hstack(y_list)
   
    y_synth = y_true + np.random.normal(0, sigma, np.size(y_true))
    return y_true, y_synth 


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('seed', metavar='s', type=int)
    args = parser.parse_args()
    seed = args.seed
    print(f'using seed: {seed}')

    ### input arguments
    model_file = "/Users/georgeau/Desktop/GitHub/BPS2023/Bayesian_Transporter/transporter_model/antiporter_15D_model_3c_0_1T.txt"
    obs_data_file = "/Users/georgeau/Desktop/GitHub/BPS2023/Bayesian_Transporter/synthetic_data/synth_data_15D_5_less_proteins.csv"
    parameter_file = "/Users/georgeau/Desktop/GitHub/BPS2023/Bayesian_Transporter/transporter_model/15D_transporter_c2_w_full_priors.json"
 
    pertubation_list = [5e-7]
    n_technical_replicas = 1
    n_exp = len(pertubation_list)

    n_walkers = 100 #1000
    n_dim = 15
    n_shuffles = 1
    near_global_min = False
    additional_samples = int(1e3) #int(1e4)
    save_every = 10
    gamma = 0.75 #0.5
    ess = 0.95 #0.99
    
    np.random.seed(seed)

    ### file i/o - create new directory, load tellurium model string, and load model parameter info
    date_string = datetime.today().strftime('%Y%m%d_%H%M%S')
    out_fname=f'run_d{date_string}_nw{n_walkers}_nd{n_dim}_as{additional_samples}_g{gamma}_ess{ess}_r{seed}'
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, out_fname)
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)
    with open(model_file, "r") as f:
        antimony_string_SS = f.read()
    with open (parameter_file, 'rb') as fp:
        p_info = json.load(fp)
    p_ref, p_labels, p_bounds = parse_p_info(p_info, near_global_min=near_global_min)
    _, _, p_bounds2 = parse_p_info(p_info, near_global_min=False)  # for plot (useful if starting near global max)

  
    # generate synthetic data
    data_label = 'synth_data_ref'
    y_true, y_obs = synthesize_data(p_ref, te.loadAntimonyModel(antimony_string_SS))
    np.savetxt(os.path.join(final_directory,f'{data_label}.csv'), y_obs, delimiter=',')
    plt.plot(y_true, label='true')
    plt.plot(y_obs, 'o', alpha=0.75, label='true+noise')
    plt.title('ion influx trace')
    plt.ylabel('influx')
    plt.xlabel('t')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(final_directory,f'{data_label}.png'))
   

    # set log likelihood arguments and initial parameter sets
    y_obs= np.genfromtxt(obs_data_file)
    p_0 = get_p0(p_bounds, n_walkers) 
    log_like_ref = calc_log_like(p_ref,y_obs,te.loada(antimony_string_SS))
    print(f"log likelihood reference: {log_like_ref}")
    # assert(1==0)
    
    # write to log file
    with open(os.path.join(final_directory, f'{out_fname}_log.txt'), "a") as f:
        f.write(f"date: {date_string}\n")
        f.write(f"model file: {model_file}\n")
        f.write(f"parameter file: {parameter_file}\n")
        f.write(f"data file: {obs_data_file}\n")
        f.write(f"seed: {seed}\n")
        f.write(f"n experiments: {n_exp}\n")
        f.write(f"pertubation list: {pertubation_list}\n")
        f.write(f"n technical replicas: {n_technical_replicas}\n")
        f.write(f"n walkers: {n_walkers}\n")
        f.write(f"n dim: {n_dim}\n")
        f.write(f"n additional samples: {additional_samples}\n")
        f.write(f"gamma: {gamma}\n")
        f.write(f"ess: {ess}\n")
        f.write(f"near global min: {near_global_min}\n")
        f.write(f"out fname: {out_fname}\n")
        f.write(f"parameter ref: {p_ref}\n")
        f.write(f"parameter labels: {p_labels}\n")
        f.write(f"parameter boundaries: {p_bounds}\n")
        f.write(f"initial parameters p_0[0]: {p_0[0]}\n")
        f.write(f"log likelihood reference: {log_like_ref}\n")

    # Number of particles to use
    n_particles = n_walkers

    # Initialise sampler
    t0 = time.time()

    sampler = pc.Sampler(
        n_walkers,
        n_dim,
        log_likelihood=calc_log_like,
        log_prior=calc_log_prior,
        vectorize_likelihood=False,
        vectorize_prior=False,
        bounds=p_bounds,
        random_state=seed,
        log_likelihood_args = [y_obs, te.loada(antimony_string_SS)],
        infer_vectorization=False,
        output_dir=final_directory,
    )

    # Start sampling
    print(f'generating initial samples from prior and starting sampler')        
    prior_samples = p_0  # Initialise particles' positions using samples from the prior (this is very important, other initialisation will not work).
    sampler.run(prior_samples, ess=ess, save_every=save_every, gamma=gamma, progress=True, n_max=(10*n_dim))

    # We can add more samples at the end
    sampler.add_samples(additional_samples)

    # Get results
    results = sampler.results  
    
    # write wall clock time to file
    wallclock = time.time() -t0
    with open(os.path.join(final_directory, f'{out_fname}_log.txt'), "a") as f:
        f.write(f"wall clock: {wallclock} sec\n")


    ### output data and plots

    # Trace plot
    pc.plotting.trace(results)   
    plt.savefig(os.path.join(final_directory,'traceplot.png'))
    plt.clf()

    # Corner plot
    pc.plotting.corner(results)
    plt.savefig(os.path.join(final_directory,'cornerplot.png'))
    plt.clf()

    # Run plot
    pc.plotting.run(results)
    plt.savefig(os.path.join(final_directory,'runplot.png'))
    plt.clf()

    np.savetxt(os.path.join(final_directory,'samples.csv'),results['samples'],delimiter=',')
    np.savetxt(os.path.join(final_directory,'loglikelihood.csv'),results['loglikelihood'],delimiter=',')
    print('sampling run complete')
