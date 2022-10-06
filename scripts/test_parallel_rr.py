import numpy as np
import tellurium as te
from pathlib import Path
import json
import ray
from ray.util import ActorPool
import matplotlib.pyplot as plt

#import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pocomc as pc 


def calc_norm_log_like_wrapper(mu,sigma,X):
    # if contains NaN --> p = 0 --> log p = -inf
    if(np.isnan(mu).any()):
      return -1e50 #-np.inf
    else:
      return calc_norm_log_like(mu,sigma,X)


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
        #m = te.loada(ms)
        m.resetToOrigin()
        m.H_out = H_out_list[i]  # plot values
        m.integrator.absolute_tolerance = 1e-18
        m.integrator.relative_tolerance = 1e-12

        # update tellurium model parameter values (rate constants)
        for key, value in k_dict.items():
            setattr(m, f'k{key}_f', 10**K[value[0]])
            setattr(m, f'k{key}_r', 10**K[value[1]])

        # k6 and k7 have cycle constraints
        m.k6_f = 10**K[10]
        m.k6_r = (m.k1_f*m.k2_f*m.k3_f*m.k4_f*m.k5_f*m.k6_f)/(m.k1_r*m.k2_r*m.k3_r*m.k4_r*m.k5_r)
        m.k7_f = 10**K[11]
        m.k7_r = (m.k2_f*m.k3_f*m.k4_f*m.k5_f*m.k7_f*m.k8_f)/(m.k2_r*m.k3_r*m.k4_r*m.k5_r*m.k8_r)

        try:
            D_tmp = m.simulate(0, 5, 125, selections=['time', 'rxn4'])
            y_tmp = D_tmp['rxn4'][1:]  # remove first point
        except:
            log_like_tmp = -np.inf  # if there is an issue calculating the flux --> no probability
            return log_like_tmp
        y_list.append(y_tmp)
    y_sim = np.hstack(y_list)
    log_like_tmp = calc_norm_log_like(y_sim,sigma,y_obs)
    return log_like_tmp


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


def get_p0(b_list, n):
    '''get initial uniform distributed samples using boundaries from b_list, and number of samples n
    b_list[i][0] = parameter lower bound, b_list[i][1] = parameter upper bound
    '''
    p0_array = np.transpose(np.array([np.random.uniform(b[0],b[1],n) for b in b_list]))  # re-arrange array for sampler
    return p0_array



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
        return -1e50#-np.inf


@ray.remote
class SimulatorActor(object):
    """Ray actor to execute simulations."""

    def __init__(self, model_string, y_obs):
        self.r = te.loada(model_string)
        self.y_obs = y_obs


    def simulate(self, K):
        """Simulate."""
        n_pts = 125
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
        

        for i in range(len(H_out_list)):
            #m = te.loada(ms)
            self.r.resetToOrigin()
            self.r.H_out = H_out_list[i]  # plot values
            self.r.integrator.absolute_tolerance = 1e-18
            self.r.integrator.relative_tolerance = 1e-12

            # update tellurium model parameter values (rate constants)
            for key, value in k_dict.items():
                setattr(self.r, f'k{key}_f', 10**K[value[0]])
                setattr(self.r, f'k{key}_r', 10**K[value[1]])

            # k6 and k7 have cycle constraints
            self.r.k6_f = 10**K[10]
            self.r.k6_r = (self.r.k1_f*self.r.k2_f*self.r.k3_f*self.r.k4_f*self.r.k5_f*self.r.k6_f)/(self.r.k1_r*self.r.k2_r*self.r.k3_r*self.r.k4_r*self.r.k5_r)
            self.r.k7_f = 10**K[11]
            self.r.k7_r = (self.r.k2_f*self.r.k3_f*self.r.k4_f*self.r.k5_f*self.r.k7_f*self.r.k8_f)/(self.r.k2_r*self.r.k3_r*self.r.k4_r*self.r.k5_r*self.r.k8_r)

            try:
                D_tmp = self.r.simulate(0, 5, n_pts, selections=['time', 'rxn4'])
                y_tmp = D_tmp['rxn4'][1:]  # remove first point
            except:
                y_tmp = np.array(n_pts * [np.NaN]) # return NaN if errors
            y_list.append(y_tmp)
        y_sim = np.hstack(y_list)
        return y_sim


    def logl(self, K):

        # print(f"logl got args {args}")
        # K = args[0]

        y_obs = self.y_obs
        y_sim = self.simulate(K)
        sigma = 10**K[-1]  # noise standard deviation is last parameter term
        return calc_norm_log_like_wrapper(y_sim, sigma, y_obs)


if __name__ == "__main__":
# ray.init(include_dashboard=False, num_cpus=2)

    ### set random seed
    seed = 0
    np.random.seed(seed)
    print(f'using seed: {seed}')

    ### input files
    # dir = Path.cwd()  # set your file path directory
    # model_file = dir / "antiporter_15D_model.txt"
    # data_file = dir / "synth_data_15D_c1_1expA_125s_v3.csv"
    # parameter_file = dir / "15D_transporter_c1_w_full_priors.json"

    model_file = "/home/groups/ZuckermanLab/georgeau/pocoMC_sampler/Bayesian_Transporter/transporter_model/antiporter_15D_model.txt"
    data_file = "/home/groups/ZuckermanLab/georgeau/pocoMC_sampler/Bayesian_Transporter/synthetic_data/synth_data_15D_c1_1expA_125s_v3.csv"
    parameter_file = "/home/groups/ZuckermanLab/georgeau/pocoMC_sampler/Bayesian_Transporter/transporter_model/15D_transporter_c1_w_full_priors.json"
    with open(model_file, "r") as f:
        model_string = f.read()
    with open (parameter_file, 'rb') as fp:
        p_info = json.load(fp)
    p_ref, p_labels, p_bounds = parse_p_info(p_info, near_global_min=False)
    _, _, p_bounds2 = parse_p_info(p_info, near_global_min=True)
    rr_model = te.loada(model_string)

    ### sampling arguments
    n_cpus = 2
    n_walkers = 10
    n_dim = 15
    additional_samples = int(1e2)
    gamma = 0.75 # (0,1) avg. correlation between walkers threshold (low-->more mcmc steps)
    ess = 0.95 # (0,1) effect sample size (~% overlap) to move to next beta value (low-->less beta stages)

    ### set log likelihood arguments and initial parameter sets
    y_obs= np.genfromtxt(data_file)
    p_0 = get_p0(p_bounds, n_walkers) 
    p_0_2 = get_p0(p_bounds2, n_walkers) 
    log_like_ref = calc_log_like(p_ref,y_obs,te.loada(model_string))
    print(f"log likelihood reference: {log_like_ref}")

    actor_count = 2  
    simulators = [SimulatorActor.remote(model_string, y_obs) for _ in range(actor_count)]

    f = lambda a, b : a.logl.remote(b)

    def calc_log_prior2(*args):
        '''calculates the log of uniform prior distribution
        if parameter is in range, the probability is log(1)-->0, else log(0)-->-inf
        '''

        p = args[0]

        # fix this later - should connect w/ parameter info data file
        b = 0  # shift parameter priors by 2 (orders of magnitude) to be non-centered
        lb = np.array([6, -1, -2, -2, 3, -1, -1, 6, -2, -2, -1, 6, -1, 3]) + b  # log10 rate constant prior lower bound + shift
        ub = np.array([12, 5, 4, 4, 9, 5, 5, 12, 4, 4, 5, 12, 5, 9]) + b  # log10 rate constant prior upper bound + shift
        sigma_lb = np.log10(5e-14)  # log10 noise sigma prior lower bound 
        sigma_ub = np.log10(5e-13)  # log10 noise sigma prior upper bound 
        if ((lb < p[:-1]) & (p[:-1] < ub)).all() and (sigma_lb<p[-1]<sigma_ub):
            return 0
        else:
            return -1e50#-np.inf

    actor_pool = ActorPool(simulators)
    sampler = pc.Sampler(
            n_walkers,
            n_dim,
            log_likelihood=f, 
            log_prior=calc_log_prior2,
            vectorize_likelihood=False,
            vectorize_prior=False,
            bounds=p_bounds,
            random_state=seed,
            infer_vectorization=False,
            log_likelihood_args = None,
            pool=actor_pool
        ) 

    sampler.log_likelihood = f
    sampler.run(p_0, ess=ess, gamma=gamma, progress=True)
    sampler.add_samples(additional_samples)
    
    results = sampler.results
    pc.plotting.corner(results)
    plt.savefig('test.png')