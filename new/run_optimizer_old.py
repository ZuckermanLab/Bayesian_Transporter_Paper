import matplotlib.pyplot as plt 
import numpy as np 
import scipy as sp
import yaml
import ssme_function as ssme
import tellurium as te
import os 
import shutil
import datetime
import logging
import time as time
from tqdm import tqdm


def get_p0(b_list, n):
    '''get initial uniform distributed samples using boundaries from b_list, and number of samples n
    b_list[i][0] = parameter lower bound, b_list[i][1] = parameter upper bound
    '''
    p0_array = np.transpose(np.array([np.random.uniform(b[0],b[1],n) for b in b_list]))  # re-arrange array for sampler
    return p0_array


def calc_normal_log_likelihood(y_obs, y_pred, sigma):
    '''calculates the log likelihood of a Normal distribution.'''
    y = sp.stats.norm.logpdf(y_obs, y_pred, sigma).sum()
    return y


def log_like(params, rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, solver_arguments):   
    k = [10**i for i in params[:-1]]
    sigma = 10**params[-1]
    try:
        res = ssme.simulate_assay(rr_model, k, initial_conditions, initial_conditions_scale, buffer_concentration_scale, solver_arguments)
        y_pred = res[1]
        log_like = calc_normal_log_likelihood(y_obs, y_pred, sigma)
        return log_like
    except:
        return -1e100


def log_like_extended(params, rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, solver_arguments):   
    k = [10**i for i in params[:-5]]
    sigma = 10**params[-1]
    bias = params[-2]
    
    # get concentration uncertainity
    H_out_buffer_scale = params[-5]
    S_out_buffer_scale = params[-4]
    initial_transporter_concentration_scale = params[-3]

    # set concentration uncertainity
    buffer_concentration_scale[0] = H_out_buffer_scale
    buffer_concentration_scale[1] = S_out_buffer_scale
    initial_conditions_scale[0] = initial_transporter_concentration_scale
    try:
        res = ssme.simulate_assay(rr_model, k, initial_conditions, initial_conditions_scale, buffer_concentration_scale, solver_arguments)
        y_pred = bias*res[1]
        log_like = calc_normal_log_likelihood(y_obs, y_pred, sigma)
        return log_like
    except:
        return -1e30


def log_prior(params, param_lb, param_ub):  
    if ((param_lb < params) & (params < param_ub)).all():
        return 0
    else:
        return -1e100
   

def log_post_wrapper(params, rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, solver_arguments, param_lb, param_ub):
    logl = log_like(params, rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, solver_arguments)
    logpr = log_prior(params, param_lb, param_ub)
    return logl+logpr


def negative_log_likelihood_wrapper(params, rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, solver_arguments, param_lb, param_ub):
    logl = log_like(params, rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, solver_arguments)
    logpr = log_prior(params, param_lb, param_ub)
    return -1*(logl+logpr)


def negative_log_likelihood_wrapper_extended(params, rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, solver_arguments, param_lb, param_ub):
    logl = log_like_extended(params, rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, solver_arguments)
    logpr = log_prior(params, param_lb, param_ub)
    return -1*(logl+logpr)


if __name__ == '__main__':

    ##### Adjust this if needed ##### 
    #config_fname = "/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/new/antiporter_1_1_12D_cycle1/experiment1/antiporter_1_1_12D_cycle1_experiment1_config.yaml"  # cycle 1 model
    #config_fname = "/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/new/antiporter_1_1_12D_cycle3/antiporter_1_1_12D_cycle3_experiment1_config.yaml"  # cycle 3 model
    #config_fname = "/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/new/antiporter_1_1_12D_cycle1/experiment1/antiporter_1_1_12D_cycle1_experiment1_config_extended.yaml"
    #config_fname = "/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/new/antiporter_1_1_12D_cycle3/antiporter_1_1_12D_cycle3_experiment1_config_extended.yaml"
    #config_fname = "/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/new/antiporter_1_1_12D_cycle2/antiporter_1_1_12D_cycle2_experiment1_config_extended.yaml"
    config_fname = "/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/new/antiporter_1_1_12D_cycle4/antiporter_1_1_12D_cycle4_experiment1_config_extended.yaml"

    ##### Setup #####
    # Load the config.yaml file
    with open(config_fname, "r") as f:
        config = yaml.safe_load(f)

    # get run configuration settings
    run_name = config["run_name"]
    model_file = config['model_file']
    data_file = config['data_file']
    simulation_kwargs = config['solver_arguments']
    inference_settings = config['bayesian_inference']
    optimization_settings = config['optimization']
    seed = config['random_seed']
    np.random.seed(seed)

    # get parameter names, values, and boundaries
    k_names = config['solver_arguments']['rate_constant_names']
    log10_p_nom = [d['nominal'] for d in config['bayesian_inference']['parameters']]
    p_names = [d['name'] for d in config['bayesian_inference']['parameters']]
    p_lb = [d['bounds'][0] for d in config['bayesian_inference']['parameters']]
    p_ub = [d['bounds'][1] for d in config['bayesian_inference']['parameters']]
    p_bounds = list(zip(p_lb,p_ub))
    p_nom = [10**d['nominal'] for d in config['bayesian_inference']['parameters']]
    k_nom = p_nom[:-1]
    sigma_nom = p_nom[-1]

    # get assay initial conditions
    initial_conditions = config['solver_arguments']['species_initial_concentrations']
    initial_conditions_scale = config['solver_arguments']['species_initial_concentrations_scale']
    buffer_concentration_scale = config['solver_arguments']['buffer_concentration_scale']

    # create new directory for runs
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
    output_fname = f'run_{run_name}_r{seed}'
    output_dir = output_fname + '_' + timestamp
    os.mkdir(output_dir)
    shutil.copy(model_file, output_dir)
    shutil.copy(data_file, output_dir)
    shutil.copy(config_fname, output_dir)

    # create logging file 
    logger = logging.getLogger('log')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(output_dir, f'log_file.log'))
    file_handler.setLevel(logging.INFO)
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    logger.info(f"output_dir: {output_dir}")
    logger.info(f"seed: {seed}")

    # load roadrunner model and simulate assay at nominal values
    rr_model = te.loadSBMLModel(model_file)
    data_nom = ssme.simulate_assay(rr_model, k_nom, initial_conditions, initial_conditions_scale, buffer_concentration_scale, simulation_kwargs)
    y_nom = data_nom[1]  # (time, output)

    # Load data, plot trace, and calculate log-likelihood reference
    y_obs = np.loadtxt(data_file, delimiter=',')
    plt.figure(figsize=(8,6))
    plt.title('Net ion influx (M/s) vs simulation step')
    plt.plot(y_nom, label='y_nom')
    plt.plot(y_obs, 'o', label='y_obs', alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'net_ion_influx_trace_nom.png'))
    log_like_nom = log_like(log10_p_nom, rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, simulation_kwargs)
    logger.info(f"log_like_nom: {log_like_nom}")

    ##### Run MLE optimization #####
    extended = config['bayesian_inference']['extended']
    n_dim = len(p_names)
    popsize = int(n_dim*2 + 1)
    logger.info(f"n_dim: {n_dim}")
    logger.info(f"popsize: {popsize}")
    if extended:
        print('using extended nll')
        optimization_func = negative_log_likelihood_wrapper_extended
    else:
        optimization_func = negative_log_likelihood_wrapper
    optimization_func_args = [rr_model, 
                                y_obs, 
                                initial_conditions, 
                                initial_conditions_scale, 
                                buffer_concentration_scale, 
                                simulation_kwargs, 
                                p_lb, 
                                p_ub
    ]
    x = []
    y = []
    s = []
    nf = []
    t = []
    n_trials = 10

    for i in tqdm(range(n_trials)):
        logger.info(f"starting optimization run {i}")
        t0 = time.time()
        res = sp.optimize.differential_evolution(optimization_func, p_bounds, args=optimization_func_args, popsize=popsize)
        dt = time.time() - t0
        logger.info(f"x: {res.x}")
        logger.info(f"func: {res.fun}")
        logger.info(f"success: {res.success}")
        logger.info(f"nfev: {res.nfev}")
        logger.info(f"time (s): {dt}\n")
        x.append(res.x)
        y.append(res.fun)
        s.append(res.success)
        nf.append(res.nfev)
        t.append(dt)
    
    ##### Save and plot the data #####
    results_list=[x,y,s,nf,t]

    y_data_list = [y,nf,t]
    fig, axes = plt.subplots(len(y_data_list), 1, figsize=(10, 12), sharex=True)
    trial_idx = [int(i+1) for i in range(n_trials)]
    y_labels = ['log-likelihood', 'n likelihood evaluations', 'runtime (s)']
    plt.suptitle('Maximum likelihood optimization runs')

    for i in range(len(y_data_list)):
        ax = axes[i]
        ax.set_ylabel(y_labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
        if y_labels[i] == 'log-likelihood':
            ax.axhline(log_like_nom, color='red')
            ax.plot(trial_idx, [-1*j for j in y_data_list[i]], '-o', )
        else:
            ax.plot(trial_idx, y_data_list[i], '-o',)
    axes[-1].set_xlabel("replica number")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'optimization_plot.png'))

    with open(os.path.join(output_dir, f'optimization_results.yaml'), 'w') as file:
        yaml.dump(results_list, file)
    #np.savetxt(os.path.join(output_dir, f'optimization_run.csv'), np.stack(results_list))
