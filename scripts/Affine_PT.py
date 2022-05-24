import numpy as np
import scipy.stats as stats, scipy as sp
import seaborn as sns
import emcee as mc
import matplotlib.pyplot as plt
import tellurium as te
import time
import corner as corner
import multiprocessing as mp
import pandas as pd

### 12d transporter

# Normal log-likelihood calculation
def calc_norm_log_likelihood(mu,sigma,X):
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


def calc_grid_point(K,y_obs,m):
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
        D_tmp = m.simulate(0, 10, 250, selections=['time', 'rxn4'])
        y_tmp = D_tmp['rxn4']
        sigma = 10**K[11]
        log_like_tmp = calc_norm_log_likelihood(y_tmp,sigma,y_obs)
    except:
        y_tmp = np.zeros(250)
        sigma = 10**K[11]
        log_like_tmp = calc_norm_log_likelihood(y_tmp,sigma,y_obs)
    return log_like_tmp


def log_prior(theta):
    '''log of uniform prior distribution'''
    
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
    

def log_prob(theta, y_obs, extra_parameters):
    '''log of estimated posterior probability'''
    m = extra_parameters[0]
    beta = extra_parameters[1]
    log_pr = log_prior(theta)
    log_l = calc_grid_point(theta, y_obs, m)
    if not np.isfinite(log_pr):
        return -np.inf  # ~zero probability
    if not np.isfinite(log_l):
        return -np.inf  # ~zero probability
    else:
        log_prob = log_pr + beta*log_l  # log posterior ~ log likelihood + log prior
        return log_prob


def set_p0():
    '''set initial walker positions'''
    
    log_k1_f = np.random.uniform(6, 12) # log10 rate constant (ref=1e10)
    log_k1_r = np.random.uniform(-1,5)  # log10 rate constant (ref=1e3) 
    log_k2_f = np.random.uniform(-2,4)  # log10 rate constant (ref=1e2)
    log_k2_r = np.random.uniform(-2,4)  # log10 rate constant (ref=1e2)

    log_k3_f = np.random.uniform(3,9)  # log10 rate constant (ref=1e7) 
    log_k3_r = np.random.uniform(-1,5)  # log10 rate constant (ref=1e3) 
    log_k4_f = np.random.uniform(-1,5)  # log10 rate constant (ref=1e3) 
    log_k4_r = np.random.uniform(6, 12)  # log10 rate constant (ref=1e10)
    log_k5_f = np.random.uniform(-2,4)  # log10 rate constant (ref=1e2)
    log_k5_r = np.random.uniform(-2,4)   # log10 rate constant (ref=1e2)
    log_k6_f = np.random.uniform(-1,5)  # log10  rate constant (ref=1e3)
    log_noise_sigma = np.random.uniform(np.log10(5e-14), np.log10(5e-13))
    
    p0_list_tmp = [        
                log_k1_f ,
                log_k1_r ,
                log_k2_f ,
                log_k2_r ,
                log_k3_f , 
                log_k3_r ,
                log_k4_f ,
                log_k4_r ,
                log_k5_f ,
                log_k5_r ,
                log_k6_f ,
                log_noise_sigma ,
    ]
    return p0_list_tmp


def plot_samples(samples, p_info, beta, run_config):
    
    n_walkers, n_steps, n_temps, seed = run_config
    ### 1D posterior plot
    samples_T = np.transpose(samples)
    fig, ax = plt.subplots(3,4, figsize=(20,15))
    axes = ax.flatten()

    for i, ax_i in enumerate(axes):
        p_tmp = p_info[i]
        print(p_tmp)
        ax_i.set_title(f"{p_tmp[0]}")
        ax_i.axvline(p_tmp[3],ls='--', color='black', alpha=0.5)
        ax_i.set_xlim(p_tmp[1],p_tmp[2])
        ax_i.hist(samples_T[i], 100, histtype="step", density=True, range=(p_tmp[1],p_tmp[2]), label=f'{p_tmp[0]}')
        ax_i.legend()

    plt.suptitle(f'1D distributions - beta={beta}')
    plt.tight_layout()
    plt.savefig(f'affine_pt_{n_walkers}w_{n_steps}s_{n_temps}t_{seed}r_dist.png') 
    

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
    
    
### model configuration
print('model configuration...')
antimony_string_SS = f"""
            // Created by libAntimony v2.12.0
            model transporter_full()

            // Compartments and Species:
            compartment vol;
            species OF in vol, OF_Hb in vol;
            species IF_Hb in vol, IF_Hb_Sb in vol;
            species IF_Sb in vol, OF_Sb in vol;
            species H_in in vol, S_in in vol;
            species $H_out in vol, $S_out in vol;

            // Reactions:
            rxn1: OF + $H_out -> OF_Hb; vol*(k1_f*OF*H_out - k1_r*OF_Hb);
            rxn2: OF_Hb -> IF_Hb; vol*(k2_f*OF_Hb - k2_r*IF_Hb);
            rxn3: IF_Hb + S_in -> IF_Hb_Sb; vol*(k3_f*IF_Hb*S_in - k3_r*IF_Hb_Sb);
            rxn4: IF_Hb_Sb -> IF_Sb + H_in; vol*(k4_f*IF_Hb_Sb - k4_r*IF_Sb*H_in);
            rxn5: IF_Sb -> OF_Sb; vol*(k5_f*IF_Sb - k5_r*OF_Sb);
            rxn6: OF_Sb -> OF + $S_out; vol*(k6_f*OF_Sb - k6_r*OF*S_out);
            

            // Events:
            E1: at (time >= 5): H_out = 1e-7, S_out = 1e-3;
            

            // Species initializations:
            H_out = 5e-8;
            H_out has substance_per_volume;

            H_in = 9.999811082242941e-08;
            H_in has substance_per_volume;

            S_out = 0.001;
            S_out has substance_per_volume;

            S_in = 0.0009999811143288836;
            S_in has substance_per_volume;

            OF = 4.7218452046117796e-09;
            OF has substance_per_volume;

            OF_Hb = 4.7218452046117796e-09;
            OF_Hb has substance_per_volume;

            IF_Hb = 4.7218452046117796e-09;
            IF_Hb has substance_per_volume;
            
            IF_Hb_Sb = 4.721756029392908e-08;
            IF_Hb_Sb has substance_per_volume;
            
            IF_Sb = 4.721845204611779e-08;
            IF_Sb has substance_per_volume;

            OF_Sb = 4.721845204611775e-08;
            OF_Sb has substance_per_volume;


            // Compartment initializations:
            vol = 0.0001;
            vol has volume;

            // Rate constant initializations:
            k1_f = 1e10;
            k1_r = 1e3;
            k2_f = 1e2;
            k2_r = 1e2;
            k3_f = 1e7;
            k3_r = 1e3;
            k4_f = 1e3;
            k4_r = 1e10;
            k5_f = 1e2;
            k5_r = 1e2;
            k6_f = 1e3;
            k6_r = 1e7;


            // Other declarations:
            const vol;
            const k1_f, k1_r, k2_f, k2_r, k3_f, k3_r;
            const k4_f, k4_r, k5_f, k5_r, k6_f, k6_r;
    

            // Unit definitions:
            unit substance_per_volume = mole / litre;
            unit volume = litre;
            unit length = metre;
            unit area = metre^2;
            unit time_unit = second;
            unit substance = mole;
            unit extent = mole;

            // Display Names:
            time_unit is "time";
            end
""" 
seed = 10
np.random.seed(seed)

m = te.loada(antimony_string_SS)
m.integrator.absolute_tolerance = 1e-18
m.integrator.relative_tolerance = 1e-12
m.H_out = 5e-8
D1 = m.simulate(0, 10, 250, selections=['time', 'rxn4'])
y_true = D1['rxn4']

noise_stdev_true = 1e-13
y_obs = np.genfromtxt("data_grid_test3_1exp.csv")

plt.figure(figsize=(10,10))
plt.plot(y_obs, 'o', alpha=0.5)
plt.plot(y_true)
plt.ylim(-2.5e-11, 2.5e-11)
plt.ylabel('y')
plt.xlabel('t')

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
print(param_ref)


# for b in beta_test:

#     make_2d_prob_grid(n_grid=100, a_range=[6,12], b_range=[-1,5], beta=b, K_true=[10,3], y_obs=y_obs, m=m )

print(f'log_prob_ref: {log_prob(param_ref, y_obs, [m,1])}')


#### sampling settings
t0 = time.time()
print(f't0 timestamp: {t0}')
print('sampling configuration...')
seed = 10
np.random.seed(seed)
dim = 12

N_steps = 100
K_walkers = 100
burn_in = 1000
M_temps = 10
NKM_samples = N_steps*K_walkers*M_temps

print(f'random seed: {seed}')
print(f'n dim: {dim}')
print(f'N steps: {N_steps}')
print(f'K walkers: {K_walkers}')
print(f'M temps: {M_temps}')
#print(f'burn in: {burn_in}')
print(f'N*K*M total samples: {NKM_samples}')
print(f'#####################')


### step 1: initialization
print('initializing...')


p0_t_list = []
for t in range(M_temps):
    p0_t_i = []
    for i in range(K_walkers):
        p0_t_i.append(set_p0())
    p0_t_list.append(p0_t_i)
p0 = np.asarray(p0_t_list)

assert(np.shape(p0) == (M_temps,K_walkers,dim))
print('sampling...')
t_0 = time.time()
# with mp.Pool(processes=2) as pool:
#     sampler = mc.PTSampler(M_temps, K_walkers, dim, calc_grid_point, log_prior, loglargs=[y_obs, m], pool=pool)
#     pos, lnprob, rstate = sampler.run_mcmc(p0, N_steps)
# samples = sampler.flatchain[0,:,:]


sampler = mc.PTSampler(M_temps, K_walkers, dim, calc_grid_point, log_prior, loglargs=[y_obs, m])
pos, lnprob, rstate = sampler.run_mcmc(p0, N_steps)
samples = sampler.flatchain[0,:,:]

lp = sampler.lnprobability

tf = time.time()
print(f'tf timestamp: {tf}')
print(f'wall clock: {tf-t0} s')
print(f'{(NKM_samples)/(tf-t0)} samples/sec' )

print(f'max sampled likelihood: {np.max(lp[0,:,:])}')
plot_samples(samples, p_info, 1, [K_walkers, N_steps, M_temps, seed])
data_df = pd.DataFrame(samples, columns=[p[0] for p in p_info])
data_df.to_csv(f'affine_pt_{K_walkers}w_{N_steps}s_{M_temps}t_{seed}r_data.csv')