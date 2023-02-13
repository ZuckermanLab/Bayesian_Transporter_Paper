import numpy as np
import tellurium as te
import matplotlib.pyplot as plt
import time
from datetime import datetime
import os
import json
import argparse
import pocomc as pc

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


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
    
    sigma = 10**K[-1]  # noise standard deviation is last parameter term
    try:
        y_sim = synthesize_assay(K, m)   
    except:
        log_like_tmp = -1e100  # if there is an issue calculating the flux --> no probability
        return log_like_tmp

    log_like_tmp = calc_norm_log_like(y_sim, sigma, y_obs)    
    return log_like_tmp


def calc_log_prior(p):
    '''calculates the log of uniform prior distribution
    if parameter is in range, the probability is log(1)-->0, else log(0)-->-inf
    '''

    # fix this later - should connect w/ parameter info data file
    b = 0  # shift parameter priors by 2 (orders of magnitude) to be non-centered
    lb = np.array([6, -1, -2, -2, 3, -1, -1, 6, -2, -2, -1, 6, -1, 3]) + b  # log10 rate constant prior lower bound + shift
    ub = np.array([12, 5, 4, 4, 9, 5, 5, 12, 4, 4, 5, 12, 5, 9]) + b  # log10 rate constant prior upper bound + shift
    sigma_lb = np.log10(5e-14)  # log10 noise sigma prior lower bound 
    sigma_ub = np.log10(5e-13)  # log10 noise sigma prior upper bound 
    if ((lb < p[:-1]) & (p[:-1] < ub)).all() and (sigma_lb<p[-1]<sigma_ub):
        return 0
    else:
        return -1e100


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


# def synthesize_assay(K,m):
#     '''synthetize a SSME assay of a transporter libroadrunner ODE model m, with parameters K
#     '''
    
#     ### assay specific
#     n_technical_replicas = 1
#     H_out_list = [5e-7]  #  experiments w/ varying external H+ concentrations [5e-7,0.2e-7]
#     t_stage = 3  # time for each stage, in seconds
#     n_pts = 60  # number of data points per stage

#     ### model specific
#     k_dict = {
#         "1":[0,1],
#         "2":[2,3],
#         "3":[4,5],
#         "4":[6,7],
#         "5":[8,9],
#         "8":[12,13]
#     }

#     sigma = 10**K[-1]  # noise standard deviation is last parameter term

#     y_list = []  # empty list to store flux from different experiments

#     ### assay
#     for i in range(len(H_out_list)):

#         # reset model and integrator tolerance
#         m.resetToOrigin()
#         m.integrator.absolute_tolerance = 1e-18
#         m.integrator.relative_tolerance = 1e-12
#         H_out_nom = m.H_out

#         # update tellurium model parameter values (rate constants)
#         for key, value in k_dict.items():
#             setattr(m, f'k{key}_f', 10**K[value[0]])
#             setattr(m, f'k{key}_r', 10**K[value[1]])

#         # rate constants k6_r and k7_r have cycle constraints (enforced in sbml model)
#         m.k6_f = 10**K[10]  
#         m.k7_f = 10**K[11]

#         # get steady values and update model
#         m.conservedMoietyAnalysis = True
#         ss_dict = dict(zip(m.steadyStateSelections, m.getSteadyStateValues()))

#         assert (np.abs(1e-3/ss_dict['[S_in]'])-np.abs(1e-7/ss_dict['[H_in]'])) < 1e-3, "equilibrium concentrations are likely incorrect"
    
#         for key, value in ss_dict.items():
#             setattr(m, f'{key}', value)
   
#         # activation 
#         m.H_out = H_out_list[i]  # set H_out activation concentration
#         D_tmp = m.simulate(0*t_stage, 1*t_stage, n_pts, selections=['time', 'current'])
#         y_tmp = D_tmp['current'][1:]  # remove first point
        
#         # relaxation 
#         m.H_out = H_out_nom  # set H_out to original concentration
#         D_tmp2 = m.simulate(1*t_stage, 2*t_stage, n_pts, selections=['time', 'current'])
#         y_tmp2 = D_tmp2['current'][1:]  # remove first point

#         y_pred = np.hstack([y_tmp, y_tmp2])
#         y_list.append(y_pred)

#     y_sim = np.hstack(y_list)

#     if n_technical_replicas > 1:
#         y_sim = np.hstack(y_list*int(n_technical_replicas))
#     return y_sim


def synthesize_assay(K,m):
    '''synthetize a SSME assay of a transporter libroadrunner ODE model m, with parameters K
    '''
    
    ### assay specific
    n_technical_replicas = 1
    H_out_list = [5e-7]  #  experiments w/ varying external H+ concentrations [5e-7,0.2e-7]
    t_stage = 3  # time for each stage, in seconds
    n_pts = 60  # number of data points per stage

    ### model specific
    k_dict = {
        "1":[0,1],
        "2":[2,3],
        "3":[4,5],
        "4":[6,7],
        "5":[8,9],
        "8":[12,13]
    }

    sigma = 10**K[-1]  # noise standard deviation is last parameter term

    y_list = []  # empty list to store flux from different experiments

    ### assay
    for i in range(len(H_out_list)):

        # reset model and integrator tolerance
        m.resetToOrigin()
        m.integrator.absolute_tolerance = 1e-18
        m.integrator.relative_tolerance = 1e-12
        H_out_nom = m.H_out

        # update tellurium model parameter values (rate constants)
        for key, value in k_dict.items():
            setattr(m, f'k{key}_f', 10**K[value[0]])
            setattr(m, f'k{key}_r', 10**K[value[1]])

        # rate constants k6_r and k7_r have cycle constraints (enforced in sbml model)
        m.k6_f = 10**K[10]  
        m.k7_f = 10**K[11]

        try:
            D_tmp = m.simulate(0, 9, 300, selections=['time', 'current'])

            #y_tmp1 = D_tmp['current'][60:120] # stage 2 with first point removed
            #y_tmp2 = D_tmp['current'][120:] # stage 3 with first point removed
            #y_tmp = np.hstack([y_tmp1, y_tmp2])
            y_tmp = D_tmp['current']
        except:
            y_tmp = np.zeros(120)

        y_list.append(y_tmp)

    if n_technical_replicas > 1:
        y_sim = np.hstack(y_list*int(n_technical_replicas))
    else:
        y_sim = np.hstack(y_list)
    return y_sim



if __name__ == "__main__":

    ### get random seed from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('seed', metavar='s', type=int)
    args = parser.parse_args()
    seed = args.seed
    print(f'using seed: {seed}')

    ### input arguments

    # exacloud
    # model_file = "/home/groups/ZuckermanLab/georgeau/pocoMC_sampler/Bayesian_Transporter/transporter_model/antiporter_15D_model.txt"
    # obs_data_file = "/home/groups/ZuckermanLab/georgeau/pocoMC_sampler/Bayesian_Transporter/synthetic_data/synth_data_15D_c1_1expA_125s_v4.csv"
    # parameter_file = "/home/groups/ZuckermanLab/georgeau/pocoMC_sampler/Bayesian_Transporter/transporter_model/15D_transporter_c1_w_full_priors.json"

    model_file = "/Users/georgeau/Desktop/GitHub/BPS2023/Bayesian_Transporter/transporter_model/15D_antiporter_model_v2.txt"
    obs_data_file = "/Users/georgeau/Desktop/GitHub/BPS2023/Bayesian_Transporter/synthetic_data/test1.csv"
    parameter_file = "/Users/georgeau/Desktop/GitHub/BPS2023/Bayesian_Transporter/transporter_model/15D_antiporter_parameters_c1_off_c2_on.json"

    pertubation_list = [5e-7]
    n_technical_replicas = 1
    n_exp = len(pertubation_list)

    n_cpus = 1
    n_walkers = 100 # 1000
    n_dim = 15
    additional_samples = int(1e2)
    save_every = 10
    gamma = 0.75  # 0.5
    ess = 0.90 # 99
    
    np.random.seed(seed)

    ### file i/o - create new directory, load tellurium model string, and load model parameter info
    date_string = datetime.today().strftime('%Y%m%d_%H%M%S')
    out_fname=f'run_poco_d{date_string}_nw{n_walkers}_nd{n_dim}_as{additional_samples}_g{gamma}_ess{ess}_r{seed}'
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, out_fname)
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)

    with open(model_file, "r") as f:
        antimony_string_SS = f.read()

    with open (parameter_file, 'rb') as fp:
        p_info = json.load(fp)

    p_ref, p_labels, p_bounds = parse_p_info(p_info, near_global_min = False)
    _, _, p_bounds2 = parse_p_info(p_info, near_global_min = False)  # for plot (useful if starting near global max)

  
    ### generate synthetic data
    y_true = synthesize_assay(p_ref, te.loadAntimonyModel(antimony_string_SS))
    y_obs = y_true + np.random.normal(loc=0, scale=10**p_ref[-1], size=np.size(y_true))
    np.savetxt(os.path.join(final_directory,'test1.csv'), y_obs, delimiter=',')
    plt.plot(y_true, label='true')
    plt.plot(y_obs, 'o', alpha=0.75, label='true+noise')
    plt.title('ion influx trace')
    plt.ylabel('influx')
    plt.xlabel('t')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(final_directory,'test.png'))
    assert(1==0)
   
    
    ### set log likelihood arguments and initial parameter sets
    y_obs = np.genfromtxt(obs_data_file)
    p_0 = get_p0(p_bounds, n_walkers) 
    log_like_ref = calc_log_like(p_ref,y_obs,te.loada(antimony_string_SS))
    print(f"log likelihood reference: {log_like_ref}")
    
    ### write to log file
    with open(os.path.join(final_directory, f'{out_fname}_log.txt'), "a") as f:
        f.write(f"date: {date_string}\n")
        f.write(f"model file: {model_file}\n")
        f.write(f"parameter file: {parameter_file}\n")
        f.write(f"data file: {obs_data_file}\n")
        f.write(f"n cpu: {n_cpus}\n")
        f.write(f"seed: {seed}\n")
        f.write(f"n experiments: {n_exp}\n")
        f.write(f"pertubation list: {pertubation_list}\n")
        f.write(f"n technical replicas: {n_technical_replicas}\n")
        f.write(f"n walkers: {n_walkers}\n")
        f.write(f"n dim: {n_dim}\n")
        f.write(f"n additional samples: {additional_samples}\n")
        f.write(f"gamma: {gamma}\n")
        f.write(f"ess: {ess}\n")
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
    sampler.run(prior_samples, ess=ess, save_every=save_every, gamma=gamma, progress=True) #, n_max=(10*n_dim))


    # We can add more samples at the end
    sampler.add_samples(additional_samples)

    # Get results
    results = sampler.results  
    

    ### write wall clock time to file
    wallclock = time.time() -t0
    with open(os.path.join(final_directory, f'{out_fname}_log.txt'), "a") as f:
        f.write(f"wall clock: {wallclock} sec\n")

    import matplotlib.pyplot as plt

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

