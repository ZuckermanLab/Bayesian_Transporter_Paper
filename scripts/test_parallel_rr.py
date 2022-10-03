import numpy as np
import tellurium as te
import roadrunner as rr
import multiprocessing as mp
import json
import time


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


#def calc_log_like(K,y_obs,m):
def calc_log_like(K,y_obs,m):
    '''calculates the log likelihood of a transporter tellurium ODE model m, given data y_obs, and parameters K
    '''
  

    H_out_list = [5e-7]  #  experiments w/ varying external pH conditions [5e-7,0.2e-7]
    #idx_list = [0,2,4,6,8]  # index of rate pairs used to set attribute, last rate omitted - fix this later 
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
        m.resetAll()
        #m.resetToOrigin()
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


if __name__ == "__main__":

    np.random.seed(0)
    n_cpus = 6
    n_sims = 1000
    model_file = "/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/transporter_model/antiporter_15D_model.txt"
    obs_data_file = "/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/synthetic_data/synth_data_15D_c1_1expA_125s_v3.csv"
    parameter_file = "/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/transporter_model/15D_transporter_c1_w_full_priors.json"

    with open(model_file, "r") as f:
        antimony_string_SS = f.read()
    rr_models = [te.loada(antimony_string_SS) for _ in range(n_sims)]
    y_obs= np.genfromtxt(obs_data_file)
    with open (parameter_file, 'rb') as fp:
        p_info = json.load(fp)
    p_ref, p_labels, p_bounds = parse_p_info(p_info, near_global_min=False)

    log_like_ref = calc_log_like(p_ref,y_obs,rr_models[0])  # testing
    print(f"log likelihood reference: {log_like_ref}")

    # create a processing pool
    p = mp.Pool(processes=n_cpus)

    # perform the simulations
    log_like_array = np.array(p.starmap(calc_log_like, [(p_ref, y_obs, rr_models[i]) for i in range(n_sims)]))
    p.close()
    print(log_like_array)
    assert(np.isclose(log_like_array.all(),log_like_ref))
    