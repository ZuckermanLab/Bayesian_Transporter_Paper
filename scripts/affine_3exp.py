import numpy as np
import scipy as sp
import emcee
import matplotlib.pyplot as plt
import tellurium as te
import time
import pandas as pd
import gc
import psutil
import ray.util.multiprocessing as mp
import ray


### 12d transporter

# Normal log-likelihood calculation
def calc_norm_log_like(mu,sigma,X):
    # Normal log-likelihood function: -[(n/2)ln(2pp*sigma^2)]-[sum((X-mu)^2)/(2*sigma^2)]
    # ref: https://www.statlect.com/fundamentals-of-statistics/normal-distribution-maximum-likelihood 
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

    sigma = 10**K[11]

    m.resetToOrigin()
    m.H_out = 5e-8
    m.integrator.absolute_tolerance = 1e-18
    m.integrator.relative_tolerance = 1e-12
    m.k1_f = 10**K[0]
    m.k1_r = 10**K[1]
    m.k2_f = 10**K[2]
    m.k2_r = 10**K[3]
    m.k3_f = 10**K[4]
    m.k3_r = 10**K[5]
    m.k4_f = 10**K[6]
    m.k4_r = 10**K[7]
    m.k5_f = 10**K[8]
    m.k5_r = 10**K[9]
    m.k6_f = 10**K[10]
    m.k6_r = (m.k1_f*m.k2_f*m.k3_f*m.k4_f*m.k5_f*m.k6_f)/(m.k1_r*m.k2_r*m.k3_r*m.k4_r*m.k5_r)
    try:
        # D_tmp = m.simulate(0, 10, 250, selections=['time', 'rxn4'])
        D_tmp1 = m.simulate(0, 5, 125, selections=['time', 'rxn4'])
        y_tmp1 = D_tmp1['rxn4'][1:]  # remove first point
        #log_like_tmp = calc_norm_log_like(y_tmp1,sigma,y_obs)
    except:
        # y_tmp = np.zeros(249)  # remove first point
        # print("error:")
        # print(K)
        # y_tmp = np.zeros(124) 
        # sigma = 10**K[11]
        # log_like_tmp = calc_norm_log_likelihood(y_tmp,sigma,y_obs)
        log_like_tmp = -np.inf
        return log_like_tmp

    m.resetToOrigin()
    m.H_out = 5e-7
    m.integrator.absolute_tolerance = 1e-18
    m.integrator.relative_tolerance = 1e-12
    m.k1_f = 10**K[0]
    m.k1_r = 10**K[1]
    m.k2_f = 10**K[2]
    m.k2_r = 10**K[3]
    m.k3_f = 10**K[4]
    m.k3_r = 10**K[5]
    m.k4_f = 10**K[6]
    m.k4_r = 10**K[7]
    m.k5_f = 10**K[8]
    m.k5_r = 10**K[9]
    m.k6_f = 10**K[10]
    m.k6_r = (m.k1_f*m.k2_f*m.k3_f*m.k4_f*m.k5_f*m.k6_f)/(m.k1_r*m.k2_r*m.k3_r*m.k4_r*m.k5_r) 
    try:
        # D_tmp = m.simulate(0, 10, 250, selections=['time', 'rxn4'])
        D_tmp2 = m.simulate(0, 5, 125, selections=['time', 'rxn4'])
        y_tmp2 = D_tmp2['rxn4'][1:]  # remove first point
        #log_like_tmp = calc_norm_log_like(y_tmp1,sigma,y_obs)
    except:
        # y_tmp = np.zeros(249)  # remove first point
        # print("error:")
        # print(K)
        # y_tmp = np.zeros(124) 
        # sigma = 10**K[11]
        # log_like_tmp = calc_norm_log_likelihood(y_tmp,sigma,y_obs)
        log_like_tmp = -np.inf
        return log_like_tmp

    m.resetToOrigin()
    m.S_out = 2e-3
    m.integrator.absolute_tolerance = 1e-18
    m.integrator.relative_tolerance = 1e-12
    m.k1_f = 10**K[0]
    m.k1_r = 10**K[1]
    m.k2_f = 10**K[2]
    m.k2_r = 10**K[3]
    m.k3_f = 10**K[4]
    m.k3_r = 10**K[5]
    m.k4_f = 10**K[6]
    m.k4_r = 10**K[7]
    m.k5_f = 10**K[8]
    m.k5_r = 10**K[9]
    m.k6_f = 10**K[10]
    m.k6_r = (m.k1_f*m.k2_f*m.k3_f*m.k4_f*m.k5_f*m.k6_f)/(m.k1_r*m.k2_r*m.k3_r*m.k4_r*m.k5_r) 
    try:
        # D_tmp = m.simulate(0, 10, 250, selections=['time', 'rxn4'])
        D_tmp3 = m.simulate(0, 5, 125, selections=['time', 'rxn4'])
        y_tmp3 = D_tmp3['rxn4'][1:]  # remove first point
        #log_like_tmp = calc_norm_log_like(y_tmp1,sigma,y_obs)
    except:
        # y_tmp = np.zeros(249)  # remove first point
        # print("error:")
        # print(K)
        # y_tmp = np.zeros(124) 
        # sigma = 10**K[11]
        # log_like_tmp = calc_norm_log_likelihood(y_tmp,sigma,y_obs)
        log_like_tmp = -np.inf
        return log_like_tmp


    y_pred = np.hstack([y_tmp1, y_tmp2, y_tmp3])
    log_like_tmp = calc_norm_log_like(y_pred,sigma,y_obs)

    return log_like_tmp


def calc_log_prior(theta):
    '''log of uniform prior distribution'''
    #p = []

    p1 = theta[0]
    p2 = theta[1]
    p3 = theta[2]
    p4 = theta[3]
    p5 = theta[4]
    p6 = theta[5]
    p7 = theta[6]
    p8 = theta[7]
    p9 = theta[8]
    p10 = theta[9]
    p11 = theta[10]
    p_sigma = theta[11]

    # if prior is between boundary --> log(prior) = 0 (uninformitive prior)
    if (np.log10(5e-14)<p_sigma<np.log10(5e-13)) and (6<p1<12) and (-1<p2<5) and (-2<p3<4) and (-2<p4<4) and \
        (3<p5<9) and (-1<p6<5) and (-1<p7<5) and (6<p8<12) and (-2<p9<4) and (-2<p10<4) and (-1<p11<5):
        return 0  
    else:
        return -np.inf
    

def calc_log_prob(theta, y_obs, extra_parameters):
    '''log of estimated posterior probability'''

    m = extra_parameters[0]
    beta = extra_parameters[1]
    log_pr = calc_log_prior(theta)
    if not np.isfinite(log_pr):
        return -np.inf  # ~zero probability
    log_like = calc_log_like(theta, y_obs, m)
    if not np.isfinite(log_like):
        return -np.inf  # ~zero probability 
    else:
        log_prob = log_pr + beta*log_like  # log posterior ~ log likelihood + log prior
        return log_prob


@ray.remote
def log_prob_ray(theta, y_obs, extra_parameters):
    return calc_log_prob(theta, y_obs, extra_parameters)

@ray.remote
def log_prob_ray_batch(batch_arg_list):
    print(batch_arg_list)
    return [calc_log_prob(arg[0], arg[1], arg[2]) for arg in batch_arg_list]


def get_p0(b_list, n):
    '''get initial samples'''
    p0_array = np.transpose(np.array([np.random.uniform(b[0],b[1],n) for b in b_list]))
    return p0_array


def plot_samples(samples, p_info, beta, run_config):
    
    n_init_samples, K_walkers, N_steps, fractional_weight_target, seed = run_config
    ### 1D posterior plot
    samples_T = np.transpose(samples)
    fig, ax = plt.subplots(3,4, figsize=(20,15))
    axes = ax.flatten()

    for i, ax_i in enumerate(axes):
        p_tmp = p_info[i]
        ax_i.set_title(f"{p_tmp[0]}")
        ax_i.axvline(p_tmp[3],ls='--', color='black', alpha=0.5)
        ax_i.set_xlim(p_tmp[1],p_tmp[2])
        ax_i.hist(samples_T[i], 100, histtype="step", density=True, range=(p_tmp[1],p_tmp[2]), label=f'{p_tmp[0]}')
        ax_i.legend()

    plt.suptitle(f'1D distributions - beta={beta}')
    plt.tight_layout()
    plt.savefig(f'affine_{K_walkers}w_{N_steps}s_{seed}r_data_3exp.png') 
    

def calculate_weights(log_like, beta_old, beta_new):
    log_w = (beta_new-beta_old)*log_like
    log_w_rel = log_w-np.max(log_w)
    w_rel = np.exp(log_w_rel)
    return w_rel
    

def calculate_next_beta(log_like, beta_old, threshold):
    def f(x):
        avg_w_rel =  np.mean(calculate_weights(log_like, beta_old, x))    
        return avg_w_rel-threshold
    
    beta_new = sp.optimize.root(f, beta_old).x[0]
    assert(beta_old <= beta_new)
    if beta_new >= 1.0:
        beta_new = 1.0
    else:
        assert(np.isclose(np.mean(calculate_weights(log_like, beta_old, beta_new)),threshold))
    return beta_new


def calculate_p_rel(log_like, beta_old, beta_new):
    w_rel = calculate_weights(log_like, beta_old, beta_new)
    mean_w_rel = np.mean(w_rel)
    p_rel = w_rel/np.sum(w_rel)
    assert(np.isclose(np.sum(p_rel),1.0))
    return p_rel
    

def auto_garbage_collect(pct=80.0):
    print(f"virtual memory before gc: {psutil.virtual_memory().percent}")
    if psutil.virtual_memory().percent >= pct:
        gc.collect()
        print(f"virtual memory after gc: {psutil.virtual_memory().percent}")


if __name__ == "__main__":    


    ### initialize parallelization using Ray
    #n_cpus=10
    # n_cpus = psutil.cpu_count(logical=False)
    # print(f"number of cpus: {n_cpus}")
    # ray.init(num_cpus=n_cpus, object_store_memory=2*10**9)
    seed = 42
    np.random.seed(seed)

    ### model configuration
    print('model configuration...')
    with open("antiporter_12D_model.txt", "r") as f:
        antimony_string_SS = f.read()

    m = te.loada(antimony_string_SS)

    # experiment 1
    m.resetToOrigin()
    m.integrator.absolute_tolerance = 1e-18
    m.integrator.relative_tolerance = 1e-12
    m.H_out = 5e-8
    D1 = m.simulate(0, 5, 125, selections=['time', 'rxn4'])
    y_true1 = D1['rxn4'][1:]  # remove first point

    # experiment 2
    m.resetToOrigin()
    m.integrator.absolute_tolerance = 1e-18
    m.integrator.relative_tolerance = 1e-12
    m.H_out = 5e-7
    D2 = m.simulate(0, 5, 125, selections=['time', 'rxn4'])
    y_true2 = D2['rxn4'][1:]  # remove first point

    # experiment 3
    m.resetToOrigin()
    m.integrator.absolute_tolerance = 1e-18
    m.integrator.relative_tolerance = 1e-12
    m.S_out = 2e-3
    D3 = m.simulate(0, 5, 125, selections=['time', 'rxn4'])
    y_true3 = D3['rxn4'][1:]  # remove first point



    y_true = np.hstack([y_true1, y_true2, y_true3])

    noise_stdev_true = 1e-13
    #y_obs = np.genfromtxt("data_grid_test3_1exp_v2.csv")  # single experiment
    #y_obs = y_true + np.random.normal(0, noise_stdev_true, np.size(y_true))
    #np.savetxt("data_grid_test3_3exp_v2.csv", y_obs)
    y_obs = np.genfromtxt("data_grid_test3_3exp_v2.csv")  # single experiment
    

    plt.figure(figsize=(10,10))
    plt.plot(y_obs, 'o', alpha=0.5)
    plt.plot(y_true)
    plt.ylim(-8.5e-11, 8.5e-11)
    plt.ylabel('y')
    plt.xlabel('t')
    plt.savefig('test2.png')
    


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
    param_ref = [p_i[3] for p_i in p_info]
    labels = [p[0] for p in p_info]
    #b_list = [(p[1], p[2]) for p in p_info]  # default
    b_list = [(p[3]*0.999, p[3]*1.001) if p[3] > 0 else (p[3]*1.001, p[3]*0.999) for p in p_info]  # near global min

    for b in b_list:
        assert(b[0]<b[1])

    print(param_ref)
    print(b_list)
 
    print(f'log_prob_ref: {calc_log_prob(param_ref, y_obs, [m,1])}')
   
    #### sampling settings
    t0 = time.time()
    print(f't0 timestamp: {t0}')
    print('sampling configuration...')
    dim = 12
    save_at = int(10)  # save data every x steps
    parallel = False
    

    N_steps = int(1e5)
    K_walkers = 100
    n_init_samples = K_walkers
    NK_total_samples = N_steps*K_walkers
    burn_in = int(0.1*NK_total_samples)

    M_sub_samples = NK_total_samples - burn_in #10*K_walkers
    assert ((M_sub_samples < NK_total_samples) & (M_sub_samples >= 2*K_walkers)), f" {M_sub_samples} < {NK_total_samples} and {M_sub_samples} >= {10*K_walkers} "
   
    print(f'#####################')
    print(f'random seed: {seed}')
    print(f'n dim: {dim}')
    print(f'n init samples: {n_init_samples}')
    print(f'N steps: {N_steps}')
    print(f'K walkers: {K_walkers}')
    print(f'M sub samples: {M_sub_samples}')
    print(f'NK total samples: {NK_total_samples}')
    print(f'burn in: {burn_in}')
    print(f'#####################')

    ### step 1: initialization
    print('initializing...')
   
    p0 = get_p0(b_list, n_init_samples)
    assert(np.shape(p0)==(n_init_samples, dim))
   
    print(f'initial walker shape: {np.shape(p0)}')
    print(f'p0: {p0[0:2]}')
    print(f"p0.size {p0.size } * samples.itemsize {p0.itemsize} = {p0.size * p0.itemsize}")
    

    if parallel==True:
        print('parallel sampling...')
        with mp.Pool() as pool:
            sampler = emcee.EnsembleSampler(K_walkers, dim, calc_log_prob, args=[y_obs, [m,1]], pool=pool)
            state = sampler.run_mcmc(p0, N_steps)
    else:
        print('serial sampling...')
        sampler = emcee.EnsembleSampler(K_walkers, dim, calc_log_prob, args=[y_obs, [m,1]])
        state = sampler.run_mcmc(p0, N_steps)

    samples = sampler.flatchain[burn_in:]

    tf = time.time()
    print(f'wall clock: {tf-t0} s')
    print(f'{(NK_total_samples)/(tf-t0)} samples/sec' )

    plot_samples(samples,p_info, 1, [n_init_samples, K_walkers, N_steps, 1, seed])
    df = pd.DataFrame(samples, columns=[i[0] for i in p_info])
    df.to_csv(f'affine_{K_walkers}w_{N_steps}s_{seed}r_data_3exp.csv')

