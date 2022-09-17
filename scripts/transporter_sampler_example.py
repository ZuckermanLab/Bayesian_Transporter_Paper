from cProfile import run
import numpy as np
import tellurium as te
import emcee 
import matplotlib.pyplot as plt
import json
import time

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
        D_tmp = m.simulate(0, 5, 50, selections=['time', 'rxn4'])
        #D_tmp = m.simulate(0, 5, 125, selections=['time', 'rxn4'])
        y_tmp = D_tmp['rxn4'][1:]  # remove first point
        sigma = 10**K[11]
        log_like_tmp = calc_norm_log_like(y_tmp,sigma,y_obs)
    except:
        log_like_tmp = -np.inf  # if there is an issue calculating the flux --> no probability
    return log_like_tmp


def calc_log_prior(p):
    '''calculates the log of uniform prior distribution
    if parameter is in range, the probability is log(1)-->0, else log(0)-->-inf
    '''
    # fix this later - should connect w/ parameter info data file
    lb = np.array([6, -1, -2, -2, 3, -1, -1, 6, -2, -2, -1])  # log10 rate constant prior lower bound + shift
    ub = np.array([12, 5, 4, 4, 9, 5, 5, 12, 4, 4, 5])  # log10 rate constant prior upper bound + shift
    sigma_lb = np.log10(5e-14)  # log10 noise sigma prior lower bound 
    sigma_ub = np.log10(5e-13)  # log10 noise sigma prior upper bound 
    #sigma_lb = np.log10(50e-14)  # log10 noise sigma prior lower bound 
    #sigma_ub = np.log10(50e-13)  # log10 noise sigma prior upper bound 

    if ((lb <= p[:-1]) & (p[:-1] <= ub)).all() and (sigma_lb<=p[-1]<=sigma_ub):
        return 0
    else:
        return -np.inf


def calc_log_post(theta, y_obs, extra_parameters):
    '''calculate the log of the posterior probability
    log posterior = log likelihood*beta + log prior^beta
    where beta is used for tempering (set = 1 usually)
    '''
    m = extra_parameters[0]
    beta = extra_parameters[1]
    log_pr = calc_log_prior(theta)
    if not np.isfinite(log_pr):
        return -np.inf # ~zero probability
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


def run_simulation(K,m):
    idx_list = [0,2,4,6,8]  # index of rate pairs used to set attribute, last rate omitted - fix this later 
    m.resetToOrigin()  # reset model concentrations, rate constants
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
    # simulate model and get flux trace (reaction 4 in this model)
    D_tmp = m.simulate(0, 5, 50, selections=['time', 'rxn4'])
    y_tmp = D_tmp['rxn4'][1:]  # remove first point
    return y_tmp



if __name__ == "__main__":

    ### input arguments
    model_file = "/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/transporter_model/antiporter_12D_model.txt"
    obs_data_file = "/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/synthetic_data/synth_data_1exp_a_trunc_50s.csv"
    parameter_file = "/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/transporter_model/12D_transporter_w_full_priors.json"
    out_fname = "test4"
    seed = 42
    n_walkers = 50
    n_dim = 12
    n_steps = 100
    thin = 1
    np.random.seed(seed)

    ### file i/o - load tellurium model string, and load model parameter info
    with open(model_file, "r") as f:
        antimony_string_SS = f.read()
    tellurium_model = te.loada(antimony_string_SS)
    with open (parameter_file, 'rb') as fp:
        p_info = json.load(fp)
    p_ref, p_labels, p_bounds = parse_p_info(p_info, near_global_min=False)
    _, _, p_bounds2 = parse_p_info(p_info, near_global_min=False)  # for plot (useful if starting near global min energy)

    ### set log likelihood arguments and initial parameter sets
    y_obs = np.genfromtxt(obs_data_file) 
    p_0 = get_p0(p_bounds, n_walkers)
    y_true = run_simulation(p_ref, tellurium_model)
    
    plt.plot(y_obs, 'o')
    plt.plot(y_true)
    plt.savefig(f'{out_fname}_flux.png')
   
  
    ### configure and run sampler (affine invariant ensemble MCMC sampler)
    ### ref: https://msp.org/camcos/2010/5-1/camcos-v5-n1-p04-p.pdf 
    sampler = emcee.EnsembleSampler(
        n_walkers, 
        n_dim, 
        calc_log_post, 
        args=[y_obs,[tellurium_model,1]],
        moves=emcee.moves.StretchMove()
        )
    t0 = time.time()
    state = sampler.run_mcmc(p_0, n_steps)
    wall_clock = time.time()-t0
    print(f'{wall_clock} s')
    print(f'{n_walkers*n_steps/wall_clock} log-likelihood calculations / s')

    ### analysis (make distribution plot and save data as .csv)
    serial_samples = sampler.get_chain(flat=True, thin=thin)
    n_bins = 100
    fig, axs = plt.subplots(4,3, figsize=(15,15))
    for i, ax in enumerate(axs.flatten()):  # for each subplot figure (parameter)  
        D_tmp = np.transpose(serial_samples)[i] 
        ax.hist(D_tmp, n_bins, histtype="step", density=True, alpha=0.85, color='k')   # plot parameter histogram
        ax.set_title(f'p_{i} distribution')
        ax.set_xlim(p_bounds2[i][0], p_bounds2[i][1])  # use full parameter range (not auto) for distribution x axis
        ax.axvline(p_ref[i], 0,1, ls='--', color='k')
        plt.tight_layout()
        plt.savefig(f'{out_fname}_distributions.png')
    np.savetxt(f"{out_fname}_samples.csv", serial_samples, delimiter=",")