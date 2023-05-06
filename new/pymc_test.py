import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import scipy as sp
import ssme_function as ssme
import os 
import yaml
import logging
import shutil
import datetime
import tellurium as te


def numerical_gradients(theta, func):
    epsilon = np.sqrt(np.finfo(float).eps)*np.ones(len(theta))
    grads = np.zeros_like(theta)
    for idx, _ in enumerate(theta):
        theta_plus = theta.copy()
        theta_plus[idx] += epsilon[idx]
        theta_minus = theta.copy()
        theta_minus[idx] -= epsilon[idx]
        grads[idx] = (func(theta_plus) - func(theta_minus)) / (2 * epsilon[idx])
    return grads



def calc_normal_log_likelihood(y_obs, y_pred, sigma):
    '''calculates the log likelihood of a Normal distribution.'''
    y = sp.stats.norm.logpdf(y_obs, y_pred, sigma).sum()
    return y


def log_like(params, rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, solver_arguments):   
    #k = [10**i for i in params[:-1]]
    sigma = 10**-7.3

    k = [10**params[0], 10**params[1], 100, 100, 10000000, 1000, 1000, 10000000000, 100, 100, 1000]

    try:
        res = ssme.simulate_assay(rr_model, k, initial_conditions, initial_conditions_scale, buffer_concentration_scale, solver_arguments)
        y_pred = res[1]
        log_like = calc_normal_log_likelihood(y_obs, y_pred, sigma)
        return log_like
    except:
        return -1e30


def approx_gradients(theta, rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, solver_arguments):
    """
    Calculate the partial derivatives of a function at a set of values.
    """

    eps = np.sqrt(np.finfo(float).eps)*np.ones(len(theta))
    #eps = 1e-8*np.ones(len(theta))
    grads = sp.optimize.approx_fprime(theta,log_like,eps,rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, solver_arguments)
    return grads


# define a pytensor Op for our likelihood function
class LogLikeWithGrad(pt.Op):

    itypes = [pt.dvector]  # expects a vector of parameter values when called
    otypes = [pt.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, simulation_kwargs):
        """
        Initialise with various things that the function requires. 
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.rr_model = rr_model
        self.y_obs = y_obs
        self.initial_conditions = initial_conditions
        self.initial_conditions_scale = initial_conditions_scale
        self.buffer_concentration_scale = buffer_concentration_scale
        self.simulation_kwargs = simulation_kwargs

        # initialise the gradient Op (below)
        self.logpgrad = LogLikeGrad(self.rr_model, self.y_obs, self.initial_conditions, 
                                 self.initial_conditions_scale, self.buffer_concentration_scale, 
                                 self.simulation_kwargs)

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        (theta,) = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(theta, self.rr_model, self.y_obs, self.initial_conditions, 
                                 self.initial_conditions_scale, self.buffer_concentration_scale, 
                                 self.simulation_kwargs)

        outputs[0][0] = np.array(logl)  # output the log-likelihood

    def grad(self, inputs, g):
        # the method that calculates the gradients - it actually returns the
        # vector-Jacobian product - g[0] is a vector of parameter values
        (theta,) = inputs  # our parameters
        return [g[0] * self.logpgrad(theta)]


class LogLikeGrad(pt.Op):

    """
    This Op will be called with a vector of values and also return a vector of
    values - the gradients in each dimension.
    """

    itypes = [pt.dvector]
    otypes = [pt.dvector]

    def __init__(self, rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, simulation_kwargs):
        """
        Initialise with various things that the function requires. 

        """

        # add inputs as class attributes
        self.rr_model = rr_model
        self.y_obs = y_obs
        self.initial_conditions = initial_conditions
        self.initial_conditions_scale = initial_conditions_scale
        self.buffer_concentration_scale = buffer_concentration_scale
        self.simulation_kwargs = simulation_kwargs


    def perform(self, node, inputs, outputs):
        (theta,) = inputs

        # calculate gradients
        grads = approx_gradients(theta,  self.rr_model, self.y_obs, self.initial_conditions, 
                                 self.initial_conditions_scale, self.buffer_concentration_scale, 
                                 self.simulation_kwargs )
        outputs[0][0] = grads


if __name__ == '__main__':
    ##### Adjust this if needed ##### 
    config_fname = "/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/new/antiporter_1_1_12D_cycle1/experiment1/antiporter_1_1_12D_cycle1_experiment1_config.yaml"  # cycle 1 model
   
    ##### Setup #####
    print(f"Running on PyMC v{pm.__version__}")

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
    k_nom = p_nom[:-1]  # check this for extended model
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
  

    # Test gradient calculation
    test_theta = np.array([10,3])
    approx_grads = approx_gradients(test_theta, rr_model, y_obs, initial_conditions,
                                     initial_conditions_scale, buffer_concentration_scale,
                                     simulation_kwargs)
    num_grads = numerical_gradients(test_theta, lambda params: log_like(test_theta, rr_model, y_obs,
                                                                        initial_conditions,
                                                                        initial_conditions_scale,
                                                                        buffer_concentration_scale,
                                                                        simulation_kwargs))
    logger.info(f"Approximated gradients: {approx_grads}")
    logger.info(f"Numerical gradients: {num_grads}" )
    logger.info(f"Gradient error: {np.abs(approx_grads - num_grads)}" )

    logger.info(f"Gradient error: {np.abs(approx_grads - num_grads)}" )

    logger.info(f"data size and shape: {np.size(y_obs)}, {np.shape(y_obs)}")
    logger.info(f"data min and max: {np.min(y_obs)}, {np.max(y_obs)}")
    logger.info(f"data mean and stdev: {np.mean(y_obs)}, {np.std(y_obs)}")

    assert(1==0)

    ##### pymc sampler configuration #####
    # create our Op
    logl = LogLikeWithGrad(log_like, rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, simulation_kwargs)

    # use PyMC to sampler from log-likelihood
    with pm.Model() as opmodel:
        # uniform priors on m and c

        log_k1_f = pm.Uniform(p_names[0], lower=p_lb[0], upper=p_ub[0])
        log_k1_r = pm.Uniform(p_names[1], lower=p_lb[1], upper=p_ub[1])
        # log_k2_f = pm.Uniform(p_names[2], lower=p_lb[2], upper=p_ub[2])
        # log_k2_r = pm.Uniform(p_names[3], lower=p_lb[3], upper=p_ub[3])
        # log_k3_f = pm.Uniform(p_names[4], lower=p_lb[4], upper=p_ub[4])
        # log_k3_r = pm.Uniform(p_names[5], lower=p_lb[5], upper=p_ub[5])
        # log_k4_f = pm.Uniform(p_names[6], lower=p_lb[6], upper=p_ub[6])
        # log_k4_r = pm.Uniform(p_names[7], lower=p_lb[7], upper=p_ub[7])
        # log_k5_f = pm.Uniform(p_names[8], lower=p_lb[8], upper=p_ub[8])
        # log_k5_r = pm.Uniform(p_names[9], lower=p_lb[9], upper=p_ub[9])
        # log_k6_f = pm.Uniform(p_names[10], lower=p_lb[10], upper=p_ub[10])
        # log_sigma = pm.Uniform(p_names[11], lower=p_lb[11], upper=p_ub[11])

        # theta_list = [
        # log_k1_f,
        # log_k1_r,
        # log_k2_f,
        # log_k2_r,
        # log_k3_f,
        # log_k3_r,
        # log_k4_f,
        # log_k4_r,
        # log_k5_f,
        # log_k5_r,
        # log_k6_f,
        # log_sigma,
        # ]

        theta_list = [log_k1_f,log_k1_r]
        # conver a tensor vector
        theta = pt.as_tensor_variable(theta_list)

        # use a Potential
        pm.Potential("likelihood", logl(theta))
        init = 'jitter+adapt_diag,'
        logger.info(f"init: {init}")
        idata_grad = pm.sample(draws=1000, tune=1000, chains=4, cores=1, step=[pm.NUTS(target_accept=0.9, max_treedepth=100)], random_seed=seed, init=init)
        #idata_grad = pm.sample(draws=1000, tune=1000, chains=4, cores=1, step=[pm.Slice()], random_seed=seed)

        # approx = pm.fit()
        # idata_grad = approx.sample(1000)
    
    

    logger.info(f"{az.summary(idata_grad, round_to=2)}")

    # plot the traces
    fig = az.plot_trace(idata_grad);
    plt.savefig(os.path.join(output_dir, f'run_plot.png'))


    # save data
    az.to_netcdf(idata_grad, os.path.join(output_dir, f'inference_data.json'))



    