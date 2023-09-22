import numpy as np 
import scipy as sp
import ssme_function as ssme
import time as time
import yaml
import tellurium as te


def get_p0(b_list, n):
    """
    Generate initial samples uniformly distributed within given boundaries.
    
    Args:
        b_list (list of lists): Each sublist contains [lower bound, upper bound] for a parameter.
        n (int): Number of samples to generate.
        
    Returns:
        ndarray: Array of initial samples.
    """
    p0_array = np.transpose(np.array([np.random.uniform(b[0],b[1],n) for b in b_list]))  # re-arrange array for sampler
    return p0_array


def calc_normal_log_likelihood(y_obs, y_pred, sigma):
    """
    Calculate the log likelihood of a Normal distribution.
    
    Args:
        y_obs (ndarray): Observed values.
        y_pred (ndarray): Predicted values.
        sigma (float): Standard deviation.
        
    Returns:
        float: Log likelihood value.
    """
    y = sp.stats.norm.logpdf(y_obs, y_pred, sigma).sum()
    return y


def log_like(params, rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, solver_arguments):  
    """
    Calculate the log likelihood for given parameters and model.
    
    Args:
        params (list): List of parameters.
         - rate constant parameters are first
         - sigma is -1 index
        rr_model (object): RoadRunner model object.
        y_obs (ndarray): Observed data.
        initial_conditions (list): List of initial species concentrations.
        initial_conditions_scale (list): List of scaling factors for the initial species concentrations.
        buffer_concentration_scale (list): List of scaling factors for the buffer concentrations.
        solver_arguments (dict): Dictionary containing the following keys:
            - roadrunner_solver_output_selections (list): Species or parameters to be included in the simulation output.
            - buffer_concentration_sequence (list of lists): Sequence of buffer concentrations for each assay stage.
            - time_per_stage (float): Duration of each assay stage.
            - n_points_per_stage (int): Number of data points per assay stage.
            - buffer_species_names (list): Names of buffer species.
            - solver_absolute_tolerance (float): Absolute tolerance for the solver.
            - solver_relative_tolerance (float): Relative tolerance for the solver.
            - remove_first_n (int): Number of initial data points to exclude from the results for each stage.
            - remove_after_n (int): Index after which to stop including data points in the results for each stage.
            - species_labels (list): Labels of species for which initial conditions are provided.
            - rate_constant_names (list): Names of rate constants in the model.
        
    Returns:
        float: Log likelihood value.
    """ 
    k = [10**i for i in params[:-1]]
    sigma = 10**params[-1]
    try:
        res = ssme.simulate_assay(rr_model, k, initial_conditions, initial_conditions_scale, buffer_concentration_scale, solver_arguments)
        y_pred = res[1]
        log_like = calc_normal_log_likelihood(y_obs, y_pred, sigma)
        return log_like
    except:
        return -1e100  # arbitrarily large negative number --> 0 probability


def log_like_extended(params, rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, solver_arguments):   
    """
    Calculate the extended log likelihood considering additional nuisance parameters - buffer solution and protein concentations, bias.
    
    Args:
        params (list): List of parameters 
         - rate constant parameters are first
         - buffer concentration scale factors (H_out and S_out) are indices -5 and -4 
         - transporter concentration scale factor is -3  index, 
         - bias factor is -2 index, 
         - sigma is -1 index
        rr_model (object): RoadRunner model object.
        y_obs (ndarray): Observed data.
        initial_conditions (list): List of initial species concentrations.
        initial_conditions_scale (list): List of scaling factors for the initial species concentrations.
        buffer_concentration_scale (list): List of scaling factors for the buffer concentrations.
        solver_arguments (dict): Dictionary containing the following keys:
            - roadrunner_solver_output_selections (list): Species or parameters to be included in the simulation output.
            - buffer_concentration_sequence (list of lists): Sequence of buffer concentrations for each assay stage.
            - time_per_stage (float): Duration of each assay stage.
            - n_points_per_stage (int): Number of data points per assay stage.
            - buffer_species_names (list): Names of buffer species.
            - solver_absolute_tolerance (float): Absolute tolerance for the solver.
            - solver_relative_tolerance (float): Relative tolerance for the solver.
            - remove_first_n (int): Number of initial data points to exclude from the results for each stage.
            - remove_after_n (int): Index after which to stop including data points in the results for each stage.
            - species_labels (list): Labels of species for which initial conditions are provided.
            - rate_constant_names (list): Names of rate constants in the model.
        
    Returns:
        float: Log likelihood value.
    """
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
        return -1e100  # arbitrarily large negative number --> 0 probability


def log_prior(params, param_lb, param_ub):  
    """
    Calculate the log prior for given parameters.
    
    Args:
        params (list): List of parameters.
        param_lb (list): Lower bounds for parameters.
        param_ub (list): Upper bounds for parameters.
        
    Returns:
        float: Log prior value.
    """
    if ((param_lb < params) & (params < param_ub)).all():
        return 0
    else:
        return -1e100  # arbitrarily large negative number --> 0 probability
   

def log_post_wrapper(params, rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, solver_arguments, param_lb, param_ub):
    """
    Wrapper function to calculate the log posterior (log likelihood + log prior).
    
    Args:
        params (list): List of parameters.
        rr_model (object): RoadRunner model object.
        y_obs (ndarray): Observed values.
        initial_conditions (list): List of initial species concentrations.
        initial_conditions_scale (list): List of scaling factors for the initial species concentrations.
        buffer_concentration_scale (list): List of scaling factors for the buffer concentrations.
        solver_arguments (dict): Dictionary containing the following keys:
            - roadrunner_solver_output_selections (list): Species or parameters to be included in the simulation output.
            - buffer_concentration_sequence (list of lists): Sequence of buffer concentrations for each assay stage.
            - time_per_stage (float): Duration of each assay stage.
            - n_points_per_stage (int): Number of data points per assay stage.
            - buffer_species_names (list): Names of buffer species.
            - solver_absolute_tolerance (float): Absolute tolerance for the solver.
            - solver_relative_tolerance (float): Relative tolerance for the solver.
            - remove_first_n (int): Number of initial data points to exclude from the results for each stage.
            - remove_after_n (int): Index after which to stop including data points in the results for each stage.
            - species_labels (list): Labels of species for which initial conditions are provided.
            - rate_constant_names (list): Names of rate constants in the model.
        param_lb (list): Lower bounds for parameters.
        param_ub (list): Upper bounds for parameters.
        
    Returns:
        float: Log posterior value.
    """
    logl = log_like(params, rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, solver_arguments)
    logpr = log_prior(params, param_lb, param_ub)
    return logl+logpr # arbitrarily large negative number --> 0 probability


def log_post_wrapper_extended(params, rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, solver_arguments, param_lb, param_ub):
    """
    Wrapper function to calculate the log posterior (log likelihood + log prior) considering considering additional nuisance parameters - buffer solution and protein concentations, bias.
    
    Args:
        params (list): List of parameters.
        rr_model (object): RoadRunner model object.
        y_obs (ndarray): Observed values.
        initial_conditions (list): List of initial species concentrations.
        initial_conditions_scale (list): List of scaling factors for the initial species concentrations.
        buffer_concentration_scale (list): List of scaling factors for the buffer concentrations.
        solver_arguments (dict): Dictionary containing the following keys:
            - roadrunner_solver_output_selections (list): Species or parameters to be included in the simulation output.
            - buffer_concentration_sequence (list of lists): Sequence of buffer concentrations for each assay stage.
            - time_per_stage (float): Duration of each assay stage.
            - n_points_per_stage (int): Number of data points per assay stage.
            - buffer_species_names (list): Names of buffer species.
            - solver_absolute_tolerance (float): Absolute tolerance for the solver.
            - solver_relative_tolerance (float): Relative tolerance for the solver.
            - remove_first_n (int): Number of initial data points to exclude from the results for each stage.
            - remove_after_n (int): Index after which to stop including data points in the results for each stage.
            - species_labels (list): Labels of species for which initial conditions are provided.
            - rate_constant_names (list): Names of rate constants in the model.
        param_lb (list): Lower bounds for parameters.
        param_ub (list): Upper bounds for parameters.
        
    Returns:
        float: Log posterior value.
    """

    logl = log_like_extended(params, rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, solver_arguments)
    logpr = log_prior(params, param_lb, param_ub)
    return logl+logpr


def negative_log_likelihood_wrapper(params, rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, solver_arguments, param_lb, param_ub):
    """
    Wrapper function to calculate the negative log likelihood.
    Technically the negative log-posterior, but the prior is uninformative so it is the same as the log-likelihood with a boundary constraint.
    
    Args:
        params (list): List of parameters.
        rr_model (object): RoadRunner model object.
        y_obs (ndarray): Observed values.
        initial_conditions (list): List of initial species concentrations.
        initial_conditions_scale (list): List of scaling factors for the initial species concentrations.
        buffer_concentration_scale (list): List of scaling factors for the buffer concentrations.
        solver_arguments (dict): Dictionary containing the following keys:
            - roadrunner_solver_output_selections (list): Species or parameters to be included in the simulation output.
            - buffer_concentration_sequence (list of lists): Sequence of buffer concentrations for each assay stage.
            - time_per_stage (float): Duration of each assay stage.
            - n_points_per_stage (int): Number of data points per assay stage.
            - buffer_species_names (list): Names of buffer species.
            - solver_absolute_tolerance (float): Absolute tolerance for the solver.
            - solver_relative_tolerance (float): Relative tolerance for the solver.
            - remove_first_n (int): Number of initial data points to exclude from the results for each stage.
            - remove_after_n (int): Index after which to stop including data points in the results for each stage.
            - species_labels (list): Labels of species for which initial conditions are provided.
            - rate_constant_names (list): Names of rate constants in the model.
        param_lb (list): Lower bounds for parameters.
        param_ub (list): Upper bounds for parameters.
        
    Returns:
        float: Negative log likelihood value.
    """
    logl = log_like(params, rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, solver_arguments)
    logpr = log_prior(params, param_lb, param_ub)
    return -1*(logl+logpr) 


def negative_log_likelihood_wrapper_extended(params, rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, solver_arguments, param_lb, param_ub):
    """
    Wrapper function to calculate the extended negative log likelihood considering concentration uncertainty. 
    Technically the negative log-posterior, but the prior is uninformative so it is the same as the log-likelihood with a boundary constraint.
    
    Args:
        params (list): List of parameters including concentration uncertainty.
        rr_model (object): RoadRunner model object.
        y_obs (ndarray): Observed values.
        initial_conditions (list): List of initial species concentrations.
        initial_conditions_scale (list): List of scaling factors for the initial species concentrations.
        buffer_concentration_scale (list): List of scaling factors for the buffer concentrations.
        solver_arguments (dict): Dictionary containing the following keys:
            - roadrunner_solver_output_selections (list): Species or parameters to be included in the simulation output.
            - buffer_concentration_sequence (list of lists): Sequence of buffer concentrations for each assay stage.
            - time_per_stage (float): Duration of each assay stage.
            - n_points_per_stage (int): Number of data points per assay stage.
            - buffer_species_names (list): Names of buffer species.
            - solver_absolute_tolerance (float): Absolute tolerance for the solver.
            - solver_relative_tolerance (float): Relative tolerance for the solver.
            - remove_first_n (int): Number of initial data points to exclude from the results for each stage.
            - remove_after_n (int): Index after which to stop including data points in the results for each stage.
            - species_labels (list): Labels of species for which initial conditions are provided.
            - rate_constant_names (list): Names of rate constants in the model.
        param_lb (list): Lower bounds for parameters.
        param_ub (list): Upper bounds for parameters.
        
    Returns:
        float: Negative log likelihood value.
    """
    logl = log_like_extended(params, rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, solver_arguments)
    logpr = log_prior(params, param_lb, param_ub)
    return -1*(logl+logpr)


def get_ssme_pred_data_from_config(params, config_file):   
    """
    Simulates a model based on parameters and a configuration file, returning the model's predictions.

    This function sets up a RoadRunner model, adjusts its parameters based on the provided `params` 
    and the `extended` field of the configuration, then simulates the model. If the simulation 
    is successful, the function returns the simulated data. Otherwise, it returns a zero array 
    of the same length as the observational data specified in the config file.

    Args:
        params (list[float]): List of model parameters.
        config_file (str): Path to the YAML configuration file which contains model, data paths,
                           and other settings.

    Returns:
        list[float]: Predicted data from the model simulation. If simulation fails, returns 
                     a zero array of the same length as the observed data.
    """

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    model_file = config['model_file']
    data_file = config['data_file']
    simulation_kwargs = config['solver_arguments']
    seed = config['random_seed']
    np.random.seed(seed)
    extended = config['bayesian_inference']['extended']
    initial_conditions = config['solver_arguments']['species_initial_concentrations']
    initial_conditions_scale = config['solver_arguments']['species_initial_concentrations_scale']
    buffer_concentration_scale = config['solver_arguments']['buffer_concentration_scale']
   
    # load roadrunner model
    rr_model = te.loadSBMLModel(model_file)

    # set parameters for ODE (SBML) model, depending on how many nuisance parameters to use
    if extended:
        # additional scaling and bias terms
        k = [10**i for i in params[:-5]]
        sigma = 10**params[-1]
        bias = params[-2]
        H_out_buffer_scale = params[-5]
        S_out_buffer_scale = params[-4]
        initial_transporter_concentration_scale = params[-3]
    else:
        k = [10**i for i in params[:-1]]
        sigma = 10**params[-1]
        bias = 1
        H_out_buffer_scale = 1
        S_out_buffer_scale = 1
        initial_transporter_concentration_scale = 1

    # set concentration uncertainity
    buffer_concentration_scale[0] = H_out_buffer_scale
    buffer_concentration_scale[1] = S_out_buffer_scale
    initial_conditions_scale[0] = initial_transporter_concentration_scale

    y_obs = np.loadtxt(data_file, delimiter=',')
    try:
        res = ssme.simulate_assay(rr_model, k, initial_conditions, initial_conditions_scale, buffer_concentration_scale, simulation_kwargs)
        y_pred = bias*res[1]
        return y_pred
    except:
        return [0]*len(y_obs)  