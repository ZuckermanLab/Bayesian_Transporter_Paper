import matplotlib.pyplot as plt 
import numpy as np 
import scipy as sp
import emcee 
import yaml
import ssme_function as ssme
import tellurium as te
import os 
import shutil
import datetime
import logging
import corner
import arviz as az


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
        return -1e30


def log_prior(params, param_lb, param_ub):  
    if ((param_lb < params) & (params < param_ub)).all():
        return 0
    else:
        return -1e30
   

def log_post_wrapper(params, rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, solver_arguments, param_lb, param_ub):
    logl = log_like(params, rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, solver_arguments)
    logpr = log_prior(params, param_lb, param_ub)
    return logl+logpr


if __name__ == '__main__':

    ##### Adjust this if needed ##### 
    config_fname = "/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/new/antiporter_1_1_12D_cycle1/experiment1/antiporter_1_1_12D_cycle1_experiment1_config.yaml"  # cycle 1 model
    #config_fname = "/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/new/antiporter_1_1_12D_cycle3/antiporter_1_1_12D_cycle3_experiment1_config.yaml"  # cycle 3 model

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

    # set sampler parameters 
    n_walkers = int(config['bayesian_inference']['mcmc_algorithm_arguments']['n_walkers'])
    n_steps = int(config['bayesian_inference']['mcmc_algorithm_arguments']['n_steps'])
    n_dim = len(p_names)
    thin = int(config['bayesian_inference']['mcmc_algorithm_arguments']['thin'])
    burn_in = int(config['bayesian_inference']['mcmc_algorithm_arguments']['burn_in'])
    logger.info(f"n_walkers: {n_walkers}")
    logger.info(f"n_steps: {n_steps}")
    logger.info(f"n_dim: {n_dim}")
    logger.info(f"thin: {thin}")
    logger.info(f"burn_in: {burn_in}")
  
    # get sampler starting points
    p_0 = get_p0(p_bounds, n_walkers) 

    ##### Run Sampler #####
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_post_wrapper, args=[rr_model, 
                                                                              y_obs, 
                                                                              initial_conditions, 
                                                                              initial_conditions_scale, 
                                                                              buffer_concentration_scale, 
                                                                              simulation_kwargs, 
                                                                              p_lb, 
                                                                              p_ub]
    )
    logger.info(f"starting MCMC")
    sampler.run_mcmc(p_0, n_steps, progress=True)
    logger.info(f"finished MCMC")

    ##### Store data #####
    samples = sampler.get_chain(discard = burn_in, thin=thin)
    flat_samples = sampler.get_chain(discard=burn_in, thin=thin, flat=True)
    log_likelihoods = sampler.get_log_prob(discard = burn_in, thin=thin)
    flat_log_likelihoods = sampler.get_log_prob(discard = burn_in, thin=thin, flat=True)
    np.savetxt(os.path.join(output_dir, f'flat_samples.csv'), flat_samples, delimiter=',')
    np.savetxt(os.path.join(output_dir, f'flat_log_likelihoods.csv'), flat_log_likelihoods, delimiter=',')
    logger.info(f"saved samples and log likelihoods")

    ##### Preliminary data viz #####
    # MCMC trace plot
    fig, axes = plt.subplots(n_dim, 1, figsize=(10, 12+(0.25*n_dim)), sharex=True)
    labels = p_names
    plt.suptitle('MCMC sampling trace')
    for i in range(n_dim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        #ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.axhline(log10_p_nom[i], color='red')
    axes[-1].set_xlabel("step number")
    plt.savefig(os.path.join(output_dir, f'mcmc_trace.png'))

    # corner plot 
    fig = corner.corner(
        flat_samples, labels=p_names, truths=log10_p_nom, range=p_bounds,
    )
    plt.savefig(os.path.join(output_dir, f'mcmc_corner.png'))

    # flux prediction plot
    plt.figure(figsize=(10,8))
    inds = np.random.randint(len(flat_samples), size=100)
    for ind in inds:
        p_sample = flat_samples[ind]
        k_tmp = [10**i for i in p_sample[:-1]]
        res_tmp = ssme.simulate_assay(rr_model, k_tmp, initial_conditions, initial_conditions_scale, buffer_concentration_scale, simulation_kwargs)
        y_tmp = res_tmp[1]
        plt.plot(y_tmp, color='blue', alpha=0.1)
    plt.plot(y_tmp, color='blue', alpha=0.1, label='y_pred')
    #plt.plot(y_nom, label='y_true', color='black')
    plt.plot(y_obs, 'o', label='y_obs', alpha=0.3,color='orange')
    plt.legend(fontsize=14)
    plt.xlabel("time step")
    plt.ylabel("net ion influx (M/s)");
    plt.savefig(os.path.join(output_dir, f'net_ion_influx_pred.png'))
    logger.info(f"plotted analysis")


