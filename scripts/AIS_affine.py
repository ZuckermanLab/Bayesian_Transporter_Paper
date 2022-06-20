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
#import multiprocessing as mp
import ray
import copy


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


def calc_log_like(K,y_obs,m):  #calc_log_like(K,y_obs,m):
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
        D_tmp = m.simulate(0, 5, 125, selections=['time', 'rxn4'])
        y_tmp = D_tmp['rxn4'][1:]  # remove first point
        sigma = 10**K[11]
        log_like_tmp = calc_norm_log_like(y_tmp,sigma,y_obs)
    except:
        # y_tmp = np.zeros(249)  # remove first point
        # print("error:")
        # print(K)
        # y_tmp = np.zeros(124) 
        # sigma = 10**K[11]
        # log_like_tmp = calc_norm_log_likelihood(y_tmp,sigma,y_obs)
        log_like_tmp = -np.inf
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

    #m = extra_parameters[0]
    #m_copy = copy.deepcopy(extra_parameters[0])
    #m_copy = extra_parameters[0]
    m = copy.copy(extra_parameters[0])
    beta = extra_parameters[1]
    log_pr = calc_log_prior(theta)
    if not np.isfinite(log_pr):
        return -np.inf  # ~zero probability
    log_like = calc_log_like(theta, y_obs, m)

    #log_like = calc_log_like(theta, y_obs, m_copy)
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


def get_p0(b_list, n, rng):
    '''get initial samples'''
    p0_array = np.transpose(np.array([rng.uniform(b[0],b[1], int(n)) for b in b_list]))
    return p0_array

@ray.remote
def get_p0_ray(b_list, n, rng):
    '''get initial samples'''
    p0_array = np.transpose(np.array([rng.uniform(b[0],b[1],n) for b in b_list]))
    return p0_array


def get_p0_batch(b_list, n, init_batch_size, rng):

    batch_list = [get_p0(b_list, init_batch_size, rng) for i in range(0, n, init_batch_size)]
    p0_array = np.vstack(batch_list)
    return p0_array
    

def get_p0_batch_ray(b_list, n, init_batch_size, rng):
    batch_list_refs = [get_p0_ray.remote(b_list, init_batch_size, rng) for i in range(0, n, init_batch_size)]
    p0_array = np.vstack(ray.get(batch_list_refs))
    return p0_array


def plot_samples(samples, p_info, beta, run_config):
    
    n_init_samples, K_walkers, N_steps, fractional_weight_target, seed, ver = run_config
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
    plt.savefig(f'TEST_ais_affine_{n_init_samples}i_{K_walkers}w_{N_steps}s_{fractional_weight_target}t_{seed}r_dist_{ver}.png') 
    

def calculate_weights(log_like, beta_old, beta_new):
    log_w = (beta_new-beta_old)*log_like
    log_w_rel = log_w-np.max(log_w)
    w_rel = np.exp(log_w_rel)
    return w_rel
    

def calculate_next_beta(log_like, beta_old, threshold):
    def f(x):
        avg_w_rel =  np.mean(calculate_weights(log_like, beta_old, x))    
        return avg_w_rel-threshold
    
    # recombine batch weights
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


    ##### initialize parallelization using Ray
    n_cpus=int(20)
    #n_cpus = psutil.cpu_count(logical=False)
    print(f"number of cpus: {n_cpus}")

    seed = 42
    rng = np.random.default_rng(seed)
    #np.random.seed(seed)
    #ray.init(num_cpus=n_cpus, object_store_memory=2*10**9)
    ##### -----------------------------------


    ##### model configuration
    print('model configuration...')
    with open("antiporter_12D_model.txt", "r") as f:
        antimony_string_SS = f.read()

   
    m = te.loada(antimony_string_SS)
    m.integrator.absolute_tolerance = 1e-18
    m.integrator.relative_tolerance = 1e-12
    m.H_out = 5e-8

    D1 = m.simulate(0, 5, 125, selections=['time', 'rxn4'])
    y_true = D1['rxn4'][1:]  # remove first point

    noise_stdev_true = 1e-13
    y_obs = np.genfromtxt("data_grid_test3_1exp_v2.csv")
    # y_obs = y_true + np.random.normal(0, noise_stdev_true, np.size(y_true))
    # np.savetxt("data_grid_test3_1exp_v2.csv", y_obs)

    plt.figure(figsize=(10,10))
    plt.plot(y_obs, 'o', alpha=0.5)
    plt.plot(y_true)
    plt.ylim(-2.5e-11, 2.5e-11)
    plt.ylabel('y')
    plt.xlabel('t')
    plt.savefig('test.png')

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
    b_list = [(p[1], p[2]) for p in p_info]

    for b in b_list:
        assert(b[0]<b[1])
    print(param_ref)
    print(b_list)
    print(f'log_prob_ref: {calc_log_prob(param_ref, y_obs, [m,1])}')
    ##### -----------------------------------


    ##### sampling settings
    t0 = time.time()
    print(f't0 timestamp: {t0}')
    print('sampling configuration...')
    dim = 12
    max_iter = 1e5
    save_at = int(10)  # save data every x steps
    
    fractional_weight_target = 0.65
    n_init_samples = int(1e6)
    N_steps = int(1e2)
    K_walkers = int(1e4)
    NK_total_samples = N_steps*K_walkers
    burn_in = int(0.1*NK_total_samples)
    init_beta_jump_target = 0.0001
    #init_batch_size = int(n_init_samples/n_cpus)
    init_batch_size = int(n_init_samples*0.1)
    #init_ESS = K_walkers
    init_fractional_weight_target = K_walkers/n_init_samples

    M_sub_samples = NK_total_samples - burn_in #10*K_walkers
    assert ((M_sub_samples < NK_total_samples) & (M_sub_samples >= 2*K_walkers)), f" {M_sub_samples} < {NK_total_samples} and {M_sub_samples} >= {2*K_walkers} "
   
    print(f'#####################')
    print(f'random seed: {seed}')
    print(f'n dim: {dim}')
    print(f'max iterations: {max_iter}')
    print(f'n init samples: {n_init_samples}')
    print(f'init beta jump target: {init_beta_jump_target}')
    print(f'init fractional weight target: {init_fractional_weight_target}')
    print(f'fractional weight target: {fractional_weight_target}')
    print(f'N steps: {N_steps}')
    print(f'K walkers: {K_walkers}')
    print(f'M sub samples: {M_sub_samples}')
    print(f'NK total samples: {NK_total_samples}')
    print(f'burn in: {burn_in}')
    print(f'initial batch size: {init_batch_size}')
    print(f'#####################')
    ##### -----------------------------------


    ##### initialization at beta=0
    print('initializing...')
    beta=0.0
    t0_2 = time.time()
    samples = get_p0(b_list, n_init_samples, rng)
    assert(np.shape(samples)==(n_init_samples, dim))
    tf_2 = time.time()
    print(f'initial walker shape: {np.shape(samples)}')
    print(f'p0: {samples[0:2]}')
    print(f"samples.size {samples.size } * samples.itemsize {samples.itemsize} = {samples.size * samples.itemsize}")
    print(f"initial uniform sampling wall clock: {tf_2 - t0_2}")
    ##### -----------------------------------


    ##### sampling routine
    print('sampling...')
    iter_i = 0
    while (beta <1) and (iter_i < max_iter):
    
        if iter_i ==0:
            print("beta=0 calculating log likelihood and weights")
            t0_3 = time.time()
            log_like = np.nan_to_num(np.array([calc_log_prob(theta_i,y_obs,[m,1]) for theta_i in samples]))
            assert(np.isnan(log_like).any()==False and np.isinf(log_like).any()==False) # no NaN or Inf
            beta_new = calculate_next_beta(log_like, beta, init_fractional_weight_target)
            p_rel = calculate_p_rel(log_like, beta, beta_new)
            tf_3 = time.time()
            print(f"calculating log likelihood and weights: wall clock for {len(samples)} = {tf_3 - t0_3}s")
            print(f"beta_new={beta_new}")
            assert(beta_new>=init_beta_jump_target)   
        else:
            print("calculating log likelihood and weights")
            t0_3 = time.time()
            log_like = np.nan_to_num(np.array([calc_log_prob(theta_i,y_obs,[m,1]) for theta_i in samples]))
            assert(np.isnan(log_like).any()==False and np.isinf(log_like).any()==False) # no NaN or Inf
            beta_new = calculate_next_beta(log_like, beta, fractional_weight_target)
            p_rel = calculate_p_rel(log_like, beta, beta_new)
            tf_3 = time.time()
            print(f"calculating log likelihood and weights: wall clock for {len(samples)} = {tf_3 - t0_3}s")
        
        print('resampling')
        t_tmp = time.time()
        resamples_index = rng.choice([ i for i in range(len(samples))],size=K_walkers,p=p_rel)
        resamples = samples[resamples_index]
        p0 = np.array([s for s in resamples])
        assert(np.shape(p0)==(K_walkers, dim))
        print(f'resampling wall clock: {time.time()-t_tmp}')

        print('running affine sampler')
        t_tmp = time.time()
        
        with mp.Pool(processes=n_cpus) as pool:
            sampler = emcee.EnsembleSampler(K_walkers, dim, calc_log_prob, args=[y_obs, [m,beta_new]], pool=pool)
            state = sampler.run_mcmc(p0, N_steps)
        #sampler = emcee.EnsembleSampler(K_walkers, dim, calc_log_prob, args=[y_obs, [m,beta_new]])
        #state = sampler.run_mcmc(p0, N_steps)

        samples = sampler.flatchain[burn_in:]
        print(f'i={iter_i}, new beta={beta_new:.2g}, old beta={beta:.2g}, delta beta={(beta_new-beta):.2g}')
        print(f"sampling wall clock: {time.time()-t_tmp}s")
  
        if iter_i%save_at == 0:
            print('saving current samples and beta...')
            np.savetxt(f'ais_affine_{n_init_samples}i_{K_walkers}w_{N_steps}s_{fractional_weight_target}t_{seed}r_data_tmp.csv',
            samples, delimiter=',', header=",".join(labels), comments=f'# i={iter_i}, beta={beta_new} ')

        iter_i = iter_i +1
        beta=beta_new
    ##### -----------------------------------


    ##### results
    tf = time.time()
    print(f'{iter_i} beta steps')
    print(f'wall clock: {tf-t0} s')
    print(f'{(iter_i*NK_total_samples)/(tf-t0)} samples/sec' )

    plot_samples(samples,p_info, beta_new, [n_init_samples, K_walkers, N_steps, fractional_weight_target, seed, "1"])
    df = pd.DataFrame(samples, columns=[i[0] for i in p_info])
    df.to_csv(f'ais_affine_{n_init_samples}i_{K_walkers}w_{N_steps}s_{fractional_weight_target}t_{seed}r_data.csv')
    #ray.shutdown()
    ##### -----------------------------------