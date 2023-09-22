import matplotlib.pyplot as plt 
import numpy as np 
import scipy as sp
import pocomc as pmc 
import yaml
import ssme_function as ssme
import tellurium as te
import os 
import shutil
import datetime
import logging
import corner
import arviz as az
import os 
import utility_functions as uf


os.environ['KMP_DUPLICATE_LIB_OK']='True'  # may be needed depending on installation of math library


def run_sampler(config_fname):
    """
    Run a Bayesian sampler (pocoMC) based on a configuration file to infer model parameters.
    
    This function sets up and runs a Bayesian sampler (MCMC) based on the provided configuration file.
    The sampler is used to infer model parameters from observed data. The function also provides
    preliminary data visualization of the MCMC results, including trace plots, corner plots, and 
    flux prediction plots. Additionally, it calculates the log evidence using both Sequential Monte Carlo 
    and Bridge Sampling methods.
    
    Args:
        config_fname (str): Path to the configuration YAML file. This file should contain all 
                            necessary settings and parameters for the run, including model file, 
                            data file, solver arguments, Bayesian inference settings, optimization 
                            settings, and random seed.
                            
    Outputs:
        - Creates a new directory for the run, containing:
            - Copied model, data, and configuration files.
            - Log file with run details.
            - MCMC run and trace plot.
            - MCMC corner plot.
            - Flux prediction plot.
            - Flat samples and log likelihoods as CSV files.
            
    Notes:
        - The configuration file should be structured appropriately with all necessary fields.
        - The function uses the `pmc` library for MCMC sampling.
        - The function uses the `pmc` library for corner plots.
        - The function assumes the use of the RoadRunner model for simulations.
        - The function handles both standard and extended models based on the 'extended' flag in the configuration.
    """

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
    extended = config['bayesian_inference']['extended']

    if extended == False:
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
    else:
        # get parameter names, values, and boundaries
        k_names = config['solver_arguments']['rate_constant_names']
        log10_p_nom = [d['nominal'] for d in config['bayesian_inference']['parameters']]
        p_names = [d['name'] for d in config['bayesian_inference']['parameters']]
        p_lb = [d['bounds'][0] for d in config['bayesian_inference']['parameters']]
        p_ub = [d['bounds'][1] for d in config['bayesian_inference']['parameters']]
        p_bounds = list(zip(p_lb,p_ub))
        p_nom = [10**d['nominal'] for d in config['bayesian_inference']['parameters']]
        k_nom = p_nom[:-5]
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
    plt.title('Current (A) vs simulation step')
    plt.plot(y_nom, label='y_nom')
    plt.plot(y_obs, 'o', label='y_obs', alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'current_trace_nom.png'))

    if extended:
        log_like_nom = uf.log_like_extended(log10_p_nom, rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, simulation_kwargs)
    else:
        log_like_nom = uf.log_like(log10_p_nom, rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, simulation_kwargs)
    logger.info(f"log_like_nom: {log_like_nom}")
    
    # set sampler parameters 
    n_walkers = int(config['bayesian_inference']['mcmc_algorithm_arguments']['n_walkers'])
    ess = config['bayesian_inference']['mcmc_algorithm_arguments']['ess']
    gamma = config['bayesian_inference']['mcmc_algorithm_arguments']['gamma']
    additional_samples = config['bayesian_inference']['mcmc_algorithm_arguments']['additional_samples']
    n_max = config['bayesian_inference']['mcmc_algorithm_arguments']['n_max']
    n_dim = len(p_names)
    thin = int(config['bayesian_inference']['mcmc_algorithm_arguments']['thin'])
    burn_in = int(config['bayesian_inference']['mcmc_algorithm_arguments']['burn_in'])
    logger.info(f"n_walkers: {n_walkers}")
    logger.info(f"ess: {ess}")
    logger.info(f"gamma: {gamma}")
    logger.info(f"additional samples: {additional_samples}")
    logger.info(f"n_max: {n_max}")
    logger.info(f"n_dim: {n_dim}")
    logger.info(f"thin: {thin}")
    logger.info(f"burn_in: {burn_in}")
  
    # get sampler starting points
    p_0 = uf.get_p0(p_bounds, n_walkers)


    ##### Run Sampler #####

    if extended:
        sampler = pmc.Sampler(
            n_walkers,
            n_dim,
            log_likelihood=uf.log_like_extended,
            log_prior=uf.log_prior,
            log_likelihood_args= [rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, simulation_kwargs],
            log_prior_args= [p_lb, p_ub],
            infer_vectorization=False,
            bounds=np.array(p_bounds),
            random_state=seed,
            diagonal = True,
        )
    else:
        sampler = pmc.Sampler(
            n_walkers,
            n_dim,
            log_likelihood=uf.log_like,
            log_prior=uf.log_prior,
            log_likelihood_args= [rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, simulation_kwargs],
            log_prior_args= [p_lb, p_ub],
            infer_vectorization=False,
            bounds=np.array(p_bounds),
            random_state=seed,
            diagonal = True, 
        )
    logger.info(f"starting MCMC")
    sampler.run(prior_samples = p_0,
            ess = ess,
            gamma = gamma,
            n_max = n_max
           )
    logger.info(f"finished MCMC")
    sampler.add_samples(additional_samples)
    logger.info(f"finished additional MCMC")

    ##### Store data #####
    results = sampler.results
    logger.info(f"total n steps: {np.sum(results['steps'])}")

    np.savetxt(os.path.join(output_dir, f'log_evidence.csv'), results['logz'], delimiter=',')
    np.savetxt(os.path.join(output_dir, f'samples.csv'), results['samples'], delimiter=',')
    np.savetxt(os.path.join(output_dir, f'log_likelihood.csv'), results['loglikelihood'], delimiter=',')
    logger.info(f"saved data")

    ##### bridge sampling for log evidence calculation #####
    logz_bs, logz_bs_error = sampler.bridge_sampling()
    logger.info(f"logZ estimated using SMC: {results['logz'][-1]}")
    logger.info(f"logZ estimated using Bridge Sampling: {logz_bs} +- {logz_bs_error}")


    ##### Preliminary data viz #####
    # run plot 
    fig = pmc.plotting.run(results)
    plt.savefig(os.path.join(output_dir, f'run_plot.png'))
    plt.close()

    # trace plot
    fig = pmc.plotting.trace(results)
    plt.savefig(os.path.join(output_dir, f'trace_plot.png'))
    plt.close()

    # corner plot 
    fig = pmc.plotting.corner(results,labels=p_names, truths=log10_p_nom, range=p_bounds)
    plt.savefig(os.path.join(output_dir, f'mcmc_corner.png'))

    # flux prediction plot
    plt.figure(figsize=(10,8))
    inds = np.random.randint(len(results['samples']), size=100)
    for ind in inds:
        p_sample = results['samples'][ind]
        k_tmp = [10**i for i in p_sample[:-1]]
        res_tmp = ssme.simulate_assay(rr_model, k_tmp, initial_conditions, initial_conditions_scale, buffer_concentration_scale, simulation_kwargs)
        y_tmp = res_tmp[1]
        plt.plot(y_tmp, color='blue', alpha=0.1)
    plt.plot(y_tmp, color='blue', alpha=0.1, label='y_pred')
    plt.plot(y_obs, 'o', label='y_obs', alpha=0.3,color='orange')
    plt.legend(fontsize=14)
    plt.xlabel("time step")
    plt.ylabel("net ion influx (M/s)")
    plt.savefig(os.path.join(output_dir, f'net_ion_influx_pred.png'))
    logger.info(f"plotted analysis")



if __name__ == '__main__':

    ##### Adjust this if needed ##### 
    example_config = "/Users/augustgeorge/Documents/GitHub/Bayesian_Transporter_Paper/config_files/antiporter_1_1_16D_cycle1_config.yaml"
    run_sampler(example_config)
    

