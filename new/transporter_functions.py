import numpy as np 
import scipy as sp
import tellurium as te


def generate_observed_data(sbml_model_parameters, rr, ssme_experiment, sigma, seed):
    np.random.seed(seed)
    results = ssme_experiment.simulate_assay(sbml_model_parameters, rr)
    t = results[0]
    y = results[1]
    y_obs = y + np.random.normal(0, sigma, np.size(y))
    return t,y_obs


def calc_log_prior(parameters, lower_bounds, upper_bounds):
    if ((lower_bounds < parameters) & (parameters < upper_bounds)).all():
        return 0
    else:
        return -1e30  # ~0 probability
    

def calc_log_likelihood(parameters, y_obs, rr, ssme_experiment, use_extended=False):

    # parameters should have format: 
    # [sbml model parameters (i.e. rate constants), protein concentration scale (optional), buffer concentration scales (optional), bias (optional), sigma]
    n_sbml_model_parameters = len(ssme_experiment.smbl_model_parameter_names)
    sbml_model_parameters = [10**p for p in parameters[:n_sbml_model_parameters]]  # convert from log10
    bias = 1.0 # unless using extended model
    if use_extended:
        # only scaling first intial concentration (transporter concentration in 'OF'/'OF_Hb_Sb' conformation)!
        # to include all initial conditions (excluding buffer) this should be modified 
        n_initial_conditions = len(ssme_experiment.initial_conditions)
        initial_conditions_scale = [1] * n_initial_conditions
        initial_conditions_scale[0] = parameters[n_sbml_model_parameters]
        buffer_concentration_scale = parameters[n_sbml_model_parameters + 1:-2] 
        ssme_experiment.update_scales(initial_conditions_scale, buffer_concentration_scale)
        bias = sbml_model_parameters[-2]
    sigma = 10**parameters[-1]  # convert from log10

    # simulate SSME experiment
    try:
        results = ssme_experiment.simulate_assay(sbml_model_parameters,rr)
        t = results[0]
        y_pred = results[1]*bias  # scalar bias term
        log_like = sp.stats.norm.logpdf(y_obs, y_pred, sigma).sum()
        return log_like
    except:
        return -1e30  # ~0 probability
   

def calc_log_posterior(parameters, y_obs, rr, ssme_experiment, lower_bounds, upper_bounds, use_extended=False):
    log_prior = calc_log_prior(parameters, lower_bounds, upper_bounds)
    log_likelihood = calc_log_likelihood(parameters, y_obs, rr, ssme_experiment, use_extended)
    log_posterior = log_prior + log_likelihood
    return log_posterior


def calc_neg_log_posterior(parameters, y_obs, rr, ssme_experiment, lower_bounds, upper_bounds, use_extended=False):
    return -1*calc_log_posterior(parameters, y_obs, rr, ssme_experiment, lower_bounds, upper_bounds, use_extended)


def calc_neg_log_posterior_wrapper(parameters, y_obs, rr, ssme_experiment, lower_bounds, upper_bounds, use_extended=False):
    #rr = te.loadSBMLModel(sbml_file)
    return -1*calc_log_posterior(parameters, y_obs, rr, ssme_experiment, lower_bounds, upper_bounds, use_extended)


def get_p0(b_list, n):
    '''get initial uniform distributed samples using boundaries from b_list, and number of samples n
    b_list[i][0] = parameter lower bound, b_list[i][1] = parameter upper bound
    '''
    p0_array = np.transpose(np.array([np.random.uniform(b[0],b[1],n) for b in b_list]))  # re-arrange array for sampler
    return p0_array



if __name__ == '__main__':
    pass
