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
from scipy.optimize import basinhopping, dual_annealing, shgo, minimize
import inspect
import utility_functions as uf



def run_optimizer(config_fname):
    """
    Run maximum likelihood estimation based on a configuration file.
    
    This function will do both (random) hyperparameter tuning and replica maximum likelihood runs for an algorithm specified in the configuration file.
    Currently the "differential_evolution", "basinhopping", "dual_annealing", "shgo", "Nelder-Mead", "Powell", "CG", "L-BFGS-B", "COBYLA", "direct" and "SLSQP"
    methods are supported.
    
    Args:
        config_fname (str): Path to the configuration YAML file. This file should contain all 
                            necessary settings and parameters for the run, including model file, 
                            data file, solver arguments, Bayesian inference settings, optimization 
                            settings, and random seed.
                            
    Outputs:
        - Creates a new directory for the run, containing:
            - Copied model, data, and configuration files.
            - Log file with run details.
            - Flux trace plot.
            - Optimization plot.
            - Tuning results.
            - Optimization results.
            
    Notes:
        - The configuration file should be structured appropriately with all necessary fields (see example folder).
        - The function uses the `scipy` library for maximum likelihood estimation.
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
    mle_method = optimization_settings['method']
    seed = config['random_seed']
    np.random.seed(seed)
    extended = config['bayesian_inference']['extended']

    # get parameter names, values, and boundaries
    k_names = config['solver_arguments']['rate_constant_names']
    log10_p_nom = [d['nominal'] for d in config['bayesian_inference']['parameters']]
    p_names = [d['name'] for d in config['bayesian_inference']['parameters']]
    p_lb = [d['bounds'][0] for d in config['bayesian_inference']['parameters']]
    p_ub = [d['bounds'][1] for d in config['bayesian_inference']['parameters']]
    p_bounds = list(zip(p_lb,p_ub))
    p_nom = [10**d['nominal'] for d in config['bayesian_inference']['parameters']]

    if extended:
        k_nom = p_nom[:-5]
    else:
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

    if extended:
        log_like_nom = uf.log_like_extended(log10_p_nom, rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, simulation_kwargs)
    else:
        log_like_nom = uf.log_like(log10_p_nom, rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, simulation_kwargs)
    logger.info(f"log_like_nom: {log_like_nom}")

    ##### Run MLE optimization #####
    tuning = config['optimization']['tuning']
    n_dim = len(p_names)
    logger.info(f"n_dim: {n_dim}")
    if extended:
        logger.info(f"using extended nll")
        optimization_func = uf.negative_log_likelihood_wrapper_extended
    else:
        optimization_func = uf.negative_log_likelihood_wrapper

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
    n_trials = optimization_settings['n_replicas']
    n_tuning_steps = config['optimization']['n_tuning_steps']

    ### Fix this! too repetitive 
    # run hyperparameter tuning
    if tuning: 
        logger.info(f"Starting optimization hyperparameter tuning")

        initial_guess = uf.get_p0(p_bounds, 1)[0]  # use the same start point for each algorithm
        logger.info(f"Initial guess: {initial_guess}")

        best_fun_value = np.inf
        best_hyperparams = None
        best_result = None
        tuning_results = []

        for i in tqdm(range(n_tuning_steps)):
            logger.info(f"Starting optimization hyperparameter tuning run {i}")
            t0 = time.time()


            if mle_method == "differential_evolution":
                # Define possible hyperparameter ranges
                strategy_choices = ['best1bin', 'best1exp', 'rand1exp', 'randtobest1exp', 'currenttobest1exp', 'best2exp', 'rand2exp', 'randtobest1bin', 'currenttobest1bin', 'best2bin', 'rand2bin', 'rand1bin']
                popsize_range = (25, 50)
                mutation_range = (0, 1.9)
                recombination_range = (0, 1)
                tol_range = (1e-5, 1e-3)
                maxiter_range = (100, 1000)  # Define a range for maxiter

                if i == 0:
                    # Use default hyperparameters for the first iteration
                    signature = inspect.signature(sp.optimize.differential_evolution)
                    parameters = signature.parameters
                    strategy = parameters["strategy"].default
                    popsize = parameters["popsize"].default
                    mutation = parameters["mutation"].default
                    recombination = parameters["recombination"].default
                    tol = parameters["tol"].default
                    maxiter = parameters["maxiter"].default  # Get default maxiter
                else:
                    # Sample random hyperparameters
                    strategy = np.random.choice(strategy_choices)
                    popsize = np.random.randint(*popsize_range)
                    mutation = np.random.uniform(*mutation_range)
                    recombination = np.random.uniform(*recombination_range)
                    tol = np.random.uniform(*tol_range)
                    maxiter = np.random.randint(*maxiter_range)  # Sample maxiter

                # Run optimization with hyperparameters
                tune_res = sp.optimize.differential_evolution(optimization_func, p_bounds, args=(rr_model, y_obs, initial_conditions, 
                                                                                initial_conditions_scale, buffer_concentration_scale, 
                                                                                simulation_kwargs, p_lb, p_ub), 
                                                                                strategy=strategy, popsize=popsize, mutation=mutation, 
                                                                                recombination=recombination, tol=tol, maxiter=maxiter)
                

                if isinstance(mutation, tuple):
                    mutation_to_store = list(mutation)
                else:
                    mutation_to_store = mutation
                # Store results
                current_hyperparams = {
                    "strategy": str(strategy),
                    "popsize": popsize,
                    "mutation": mutation_to_store,
                    "recombination": recombination,
                    "tol": tol,
                    "maxiter": maxiter
                }

            elif mle_method == "basinhopping":
                # Define possible hyperparameter ranges for basinhopping
                niter_range = (50, 500)  # Number of basin hopping iterations
                T_range = (0.5, 5.0)  # Temperature for acceptance criterion
                stepsize_range = (0.1, 2.0)  # Step size for random displacement
                interval_range = (10, 100)  # Interval for updating the step size
                niter_success_range = (5, 50)  # Number of iterations for success check

                if i == 0:
                    # Use default hyperparameters for the first iteration
                    signature = inspect.signature(sp.optimize.basinhopping)
                    parameters = signature.parameters
                    niter = parameters["niter"].default
                    T = parameters["T"].default
                    stepsize = parameters["stepsize"].default
                    interval = parameters["interval"].default
                    niter_success = parameters["niter_success"].default
                else:
                    # Sample random hyperparameters for basinhopping
                    niter = np.random.randint(*niter_range)
                    T = np.random.uniform(*T_range)
                    stepsize = np.random.uniform(*stepsize_range)
                    interval = np.random.randint(*interval_range)
                    niter_success = np.random.randint(*niter_success_range)

                # Run optimization with hyperparameters
                minimizer_kwargs = {"args": (rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, simulation_kwargs, p_lb, p_ub), "bounds": p_bounds}
                tune_res = sp.optimize.basinhopping(optimization_func, initial_guess, minimizer_kwargs=minimizer_kwargs, niter=niter, T=T, stepsize=stepsize, interval=interval, niter_success=niter_success)

                # Store results
                current_hyperparams = {
                    "niter": niter,
                    "T": T,
                    "stepsize": stepsize,
                    "interval": interval,
                    "niter_success": niter_success
                }

            elif mle_method == "dual_annealing":
                # Define possible hyperparameter ranges
                initial_temp_range = (0.01, 50000)
                restart_temp_ratio_range = (0, 1)
                visit_range = (1, 3)
                accept_range = (-1e4, -5)
                maxiter_range = (100, 1000)
                maxfun_range = (1000, 1e7)
                no_local_search_choices = [True, False]

                if i == 0:
                    # Use default hyperparameters for the first iteration
                    signature = inspect.signature(sp.optimize.dual_annealing)
                    parameters = signature.parameters
                    initial_temp = parameters["initial_temp"].default
                    restart_temp_ratio = parameters["restart_temp_ratio"].default
                    visit = parameters["visit"].default
                    accept = parameters["accept"].default
                    maxiter = parameters["maxiter"].default
                    maxfun = parameters["maxfun"].default
                    no_local_search = parameters["no_local_search"].default
                else:
                    # Sample random hyperparameters
                    initial_temp = np.random.uniform(*initial_temp_range)
                    restart_temp_ratio = np.random.uniform(*restart_temp_ratio_range)
                    visit = np.random.uniform(*visit_range)
                    accept = np.random.uniform(*accept_range)
                    maxiter = np.random.randint(*maxiter_range)
                    maxfun = np.random.randint(*maxfun_range)
                    no_local_search = np.random.choice(no_local_search_choices)

                # Run optimization with hyperparameters
                tune_res = sp.optimize.dual_annealing(optimization_func, bounds=p_bounds, 
                                                    args=(rr_model, y_obs, initial_conditions, 
                                                            initial_conditions_scale, buffer_concentration_scale, 
                                                            simulation_kwargs, p_lb, p_ub), 
                                                    initial_temp=initial_temp, restart_temp_ratio=restart_temp_ratio, 
                                                    visit=visit, accept=accept, maxiter=maxiter, maxfun=maxfun, 
                                                    no_local_search=no_local_search)
                # Store results
                current_hyperparams = {
                    "initial_temp": initial_temp,
                    "restart_temp_ratio": restart_temp_ratio,
                    "visit": visit,
                    "accept": accept,
                    "maxiter": maxiter,
                    "maxfun": maxfun,
                    "no_local_search": bool(no_local_search)
                }

            elif mle_method == "shgo":
                # Define possible hyperparameter ranges
                n_range = (25, 500)  
                iters_range = (1, 10)  
                sampling_methods = ["simplicial", "halton", "sobol"]
                maxiter_range = (100, 1000)
                maxev_range = (100, 10000)  
                maxtime_range = (0.1, 10)  
                minimize_every_iter_choices = [True, False]
                local_iter_range = (10, 100)  
                infty_constraints_choices = [True, False]
                            
                if i == 0:
                    # Use default hyperparameters for the first iteration
                    signature = inspect.signature(sp.optimize.shgo)
                    parameters = signature.parameters
                    n = parameters["n"].default
                    iters = parameters["iters"].default
                    sampling_method = parameters["sampling_method"].default
                    
                    options_defaults = parameters["options"].default
                    if options_defaults is None:
                        options_defaults = {}
                    
                    maxiter = options_defaults.get("maxiter", maxiter_range[0])
                    maxev = options_defaults.get("maxev", maxev_range[0])
                    maxtime = options_defaults.get("maxtime", maxtime_range[0])
                    minimize_every_iter = options_defaults.get("minimize_every_iter", True)
                    local_iter = options_defaults.get("local_iter", local_iter_range[0])
                    infty_constraints = options_defaults.get("infty_constraints", True)
                else:
                    # Sample random hyperparameters
                    n = np.random.randint(*n_range)
                    iters = np.random.randint(*iters_range)
                    sampling_method = np.random.choice(sampling_methods)
                    maxiter = np.random.randint(*maxiter_range)
                    maxev = np.random.randint(*maxev_range)
                    maxtime = np.random.uniform(*maxtime_range)
                    minimize_every_iter = np.random.choice(minimize_every_iter_choices)
                    local_iter = np.random.randint(*local_iter_range)
                    infty_constraints = np.random.choice(infty_constraints_choices)

                # Run optimization with hyperparameters
                tune_res = sp.optimize.shgo(optimization_func, bounds=p_bounds, 
                                            args=(rr_model, y_obs, initial_conditions, 
                                                initial_conditions_scale, buffer_concentration_scale, 
                                                simulation_kwargs, p_lb, p_ub), 
                                            n=n, iters=iters, sampling_method=sampling_method,
                                            options={"maxiter": maxiter, "maxev": maxev, "maxtime": maxtime,
                                                    "minimize_every_iter": minimize_every_iter, "local_iter": local_iter,
                                                    "infty_constraints": infty_constraints})
                            
                # Store results
                current_hyperparams = {
                    "n": n,
                    "iters": iters,
                    "sampling_method": sampling_method,
                    "maxiter": maxiter,
                    "maxev": maxev,
                    "maxtime": maxtime,
                    "minimize_every_iter": bool(minimize_every_iter),
                    "local_iter": local_iter,
                    "infty_constraints": bool(infty_constraints)
                }

            elif mle_method == "Nelder-Mead":
                # Define possible hyperparameter ranges
                maxiter_range = (1000, 10000)
                maxfev_range = (1000, 10000)
                xatol_range = (1e-8, 1e-4)
                fatol_range = (1e-8, 1e-4)
                adaptive_choices = [True, False]

                if i == 0:  # Use default hyperparameters for the first iteration
                    # Hard-coded default values based on scipy documentation / common values
                    maxiter = len(p_bounds) * 200  # N * 200, where N is the number of variables
                    maxfev = len(p_bounds) * 200  # Same as maxiter by default
                    xatol = 0.0001  
                    fatol = 0.0001  
                    adaptive = True
                else:
                    # Sample random hyperparameters
                    maxiter = np.random.randint(*maxiter_range)
                    maxfev = np.random.randint(*maxfev_range)
                    xatol = np.random.uniform(*xatol_range)
                    fatol = np.random.uniform(*fatol_range)
                    adaptive = np.random.choice(adaptive_choices)

                # Run optimization with hyperparameters
                tune_res = sp.optimize.minimize(optimization_func, x0=initial_guess, args=(rr_model, y_obs, initial_conditions, 
                                                                              initial_conditions_scale, buffer_concentration_scale, 
                                                                              simulation_kwargs, p_lb, p_ub),
                                                method='Nelder-Mead', 
                                                options={"maxiter": maxiter, "maxfev": maxfev, "xatol": xatol, "fatol": fatol, "adaptive": adaptive})


                # Store results
                current_hyperparams = {
                    "maxiter": maxiter,
                    "maxfev": maxfev,
                    "xatol": xatol,
                    "fatol": fatol,
                    "adaptive": bool(adaptive)
                }

            elif mle_method == "Powell":
                # Define possible hyperparameter ranges
                maxiter_range = (len(p_bounds) * 1000, len(p_bounds) * 10000)
                maxfev_range = (len(p_bounds) * 1000, len(p_bounds) * 10000)
                xtol_range = (1e-8, 1e-4)
                ftol_range = (1e-8, 1e-4)

                if i == 0:  # Use default hyperparameters for the first iteration
                    # Hard-coded default values based on scipy documentation / common values
                    maxiter = len(p_bounds) * 1000  # N * 1000, where N is the number of variables
                    maxfev = len(p_bounds) * 1000  # Same as maxiter by default
                    xtol = 0.0001  # Placeholder value based on common settings
                    ftol = 0.0001  # Placeholder value based on common settings
                else:
                    # Sample random hyperparameters
                    maxiter = np.random.randint(*maxiter_range)
                    maxfev = np.random.randint(*maxfev_range)
                    xtol = np.random.uniform(*xtol_range)
                    ftol = np.random.uniform(*ftol_range)

                # Run optimization with hyperparameters
                tune_res = sp.optimize.minimize(optimization_func, x0=initial_guess, args=(rr_model, y_obs, initial_conditions, 
                                                                                        initial_conditions_scale, buffer_concentration_scale, 
                                                                                        simulation_kwargs, p_lb, p_ub),
                                                method='Powell', 
                                                options={"maxiter": maxiter, "maxfev": maxfev, "xtol": xtol, "ftol": ftol})

                # Store results
                current_hyperparams = {
                    "maxiter": maxiter,
                    "maxfev": maxfev,
                    "xtol": xtol,
                    "ftol": ftol
                }


            elif mle_method == "CG":
                # Define possible hyperparameter ranges
                maxiter_range = (len(p_bounds) * 1000, len(p_bounds) * 10000)
                gtol_range = (1e-8, 1e-4)
                norm_choices = [np.inf, -np.inf, 1, 2]  
                eps_range = (1e-8, 1e-4)  

                if i == 0:  # Use default hyperparameters for the first iteration
                    # Hard-coded default values
                    maxiter = len(p_bounds) * 1000  
                    gtol = 0.0001  
                    norm = np.inf 
                    eps = 0.0001  
                else:
                    # Sample random hyperparameters
                    maxiter = np.random.randint(*maxiter_range)
                    gtol = np.random.uniform(*gtol_range)
                    norm = np.random.choice(norm_choices)
                    eps = np.random.uniform(*eps_range)

                # Run optimization with hyperparameters
                tune_res = sp.optimize.minimize(optimization_func, x0=initial_guess, args=(rr_model, y_obs, initial_conditions, 
                                                                                    initial_conditions_scale, buffer_concentration_scale, 
                                                                                    simulation_kwargs, p_lb, p_ub),
                                                method='CG', 
                                                options={"maxiter": maxiter, "gtol": gtol, "norm": norm, "eps": eps})

                if norm == np.inf:
                    norm_to_save = float('inf')
                elif norm == -np.inf:
                    norm_to_save = float('-inf')
                else:
                    norm_to_save = float(norm)

                # Store results
                current_hyperparams = {
                    "maxiter": maxiter,
                    "gtol": gtol,
                    "norm": norm_to_save,
                    "eps": eps
                }

            elif mle_method == "L-BFGS-B":
                # Define possible hyperparameter ranges
                maxcor_range = (5, 20) 
                maxfun_range = (len(p_bounds) * 1000, len(p_bounds) * 10000)
                maxiter_range = (len(p_bounds) * 1000, len(p_bounds) * 10000)
                ftol_range = (1e-10, 1e-4)
                gtol_range = (1e-10, 1e-4)
                eps_range = (1e-10, 1e-4)
                maxls_range = (10, 100) 

                if i == 0:  # Use default hyperparameters for the first iteration
                    # Hard-coded default values
                    maxcor = 10  
                    maxfun = len(p_bounds) * 1000
                    maxiter = len(p_bounds) * 1000
                    ftol = 2.2e-9  # ftol = factr * numpy.finfo(float).eps where factr = 1e7.
                    gtol = 1e-5  
                    eps = 1e-8  
                    maxls = 20  
                else:
                    # Sample random hyperparameters
                    maxcor = np.random.randint(*maxcor_range)
                    maxfun = np.random.randint(*maxfun_range)
                    maxiter = np.random.randint(*maxiter_range)
                    ftol = np.random.uniform(*ftol_range)
                    gtol = np.random.uniform(*gtol_range)
                    eps = np.random.uniform(*eps_range)
                    maxls = np.random.randint(*maxls_range)

                # Run optimization with hyperparameters
                tune_res = sp.optimize.minimize(optimization_func, x0=initial_guess, args=(rr_model, y_obs, initial_conditions, 
                                                                                        initial_conditions_scale, buffer_concentration_scale, 
                                                                                        simulation_kwargs, p_lb, p_ub),
                                                method='L-BFGS-B', 
                                                bounds=p_bounds,  # Necessary for L-BFGS-B
                                                options={"maxcor": maxcor, "maxfun": maxfun, "maxiter": maxiter, "ftol": ftol, 
                                                        "gtol": gtol, "eps": eps, "maxls": maxls})

                # Store results
                current_hyperparams = {
                    "maxcor": maxcor,
                    "maxfun": maxfun,
                    "maxiter": maxiter,
                    "ftol": ftol,
                    "gtol": gtol,
                    "eps": eps,
                    "maxls": maxls
                }

            elif mle_method == "COBYLA":
                # Define possible hyperparameter ranges
                rhobeg_range = (0.01, 1.0)  
                tol_range = (1e-10, 1e-4)
                maxiter_range = (len(p_bounds) * 1000, len(p_bounds) * 10000)
                catol_range = (1e-10, 1e-4)  

                if i == 0:  # Use default hyperparameters for the first iteration
                    # Hard-coded default values
                    rhobeg = 0.1  
                    tol = 1e-6  
                    maxiter = len(p_bounds) * 1000
                    catol = 1e-6  
                else:
                    # Sample random hyperparameters
                    rhobeg = np.random.uniform(*rhobeg_range)
                    tol = np.random.uniform(*tol_range)
                    maxiter = np.random.randint(*maxiter_range)
                    catol = np.random.uniform(*catol_range)

                # Run optimization with hyperparameters
                tune_res = sp.optimize.minimize(optimization_func, x0=initial_guess, args=(rr_model, y_obs, initial_conditions, 
                                                                                            initial_conditions_scale, buffer_concentration_scale, 
                                                                                            simulation_kwargs, p_lb, p_ub),
                                                method='COBYLA', 
                                                options={"rhobeg": rhobeg, "tol": tol, "maxiter": maxiter, "catol": catol, "disp": False})  # disp set to True for convergence messages

                # Store results
                current_hyperparams = {
                    "rhobeg": rhobeg,
                    "tol": tol,
                    "maxiter": maxiter,
                    "catol": catol
                }


            elif mle_method == "direct":
                # Define possible hyperparameter ranges
                eps_range = (1e-6, 1e-2)
                maxfun_range = (10000, 100000)
                maxiter_range = (500, 2000)
                locally_biased_choices = [True, False]
                f_min_rtol_range = (1e-6, 1e-2)
                vol_tol_range = (1e-20, 1e-10)
                len_tol_range = (1e-8, 1e-4)

                if i == 0:  # Use default hyperparameters for the first iteration
                    # Get the default values from the sp.optimize.direct function
                    signature = inspect.signature(sp.optimize.direct)
                    parameters = signature.parameters

                    default_eps = parameters["eps"].default
                    default_maxfun = parameters["maxfun"].default
                    default_maxiter = parameters["maxiter"].default
                    default_locally_biased = parameters["locally_biased"].default
                    default_f_min_rtol = parameters["f_min_rtol"].default
                    default_vol_tol = parameters["vol_tol"].default
                    default_len_tol = parameters["len_tol"].default
                    eps = default_eps
                    maxfun = default_maxfun
                    maxiter = default_maxiter
                    locally_biased = default_locally_biased
                    f_min_rtol = default_f_min_rtol
                    vol_tol = default_vol_tol
                    len_tol = default_len_tol
                else:
                    # Sample random hyperparameters
                    eps = np.random.uniform(*eps_range)
                    maxfun = np.random.randint(*maxfun_range)
                    maxiter = np.random.randint(*maxiter_range)
                    locally_biased = np.random.choice(locally_biased_choices)
                    f_min_rtol = np.random.uniform(*f_min_rtol_range)
                    vol_tol = np.random.uniform(*vol_tol_range)
                    len_tol = np.random.uniform(*len_tol_range)

                # Run optimization with hyperparameters
                tune_res = sp.optimize.direct(optimization_func, p_bounds, args=(rr_model, y_obs, initial_conditions, 
                                                                                initial_conditions_scale, buffer_concentration_scale, 
                                                                                simulation_kwargs, p_lb, p_ub),
                                            eps=eps, maxfun=maxfun, maxiter=maxiter, locally_biased=bool(locally_biased),
                                            f_min_rtol=f_min_rtol, vol_tol=vol_tol, len_tol=len_tol)

                # Store results
                current_hyperparams = {
                    "eps": eps,
                    "maxfun": maxfun,
                    "maxiter": maxiter,
                    "locally_biased": bool(locally_biased),
                    "f_min_rtol": f_min_rtol,
                    "vol_tol": vol_tol,
                    "len_tol": len_tol
                }

            elif mle_method == "SLSQP":
                # Define possible hyperparameter ranges
                ftol_range = (1e-10, 1e-4)
                eps_range = (1e-10, 1e-4)
                maxiter_range = (len(p_bounds) * 1000, len(p_bounds) * 10000)

                if i == 0:  # Use default hyperparameters for the first iteration
                    # Hard-coded default values
                    ftol = 1e-6
                    eps = 1e-8
                    maxiter = len(p_bounds) * 1000
          
                else:
                    # Sample random hyperparameters
                    ftol = np.random.uniform(*ftol_range)
                    eps = np.random.uniform(*eps_range)
                    maxiter = np.random.randint(*maxiter_range)

                # Run optimization with hyperparameters
                tune_res = sp.optimize.minimize(optimization_func, x0=initial_guess, args=(rr_model, y_obs, initial_conditions, 
                                                                                            initial_conditions_scale, buffer_concentration_scale, 
                                                                                            simulation_kwargs, p_lb, p_ub),
                                                method='SLSQP', 
                                                bounds=p_bounds,  
                                                options={"ftol": ftol, "eps": eps, "maxiter": maxiter, "disp": False,})

                # Store results
                current_hyperparams = {
                    "ftol": ftol,
                    "eps": eps,
                    "maxiter": maxiter,
                }

            else:
                print('error: invalid MLE algorithm.')
                assert(1==0)

            tuning_results.append({
                "hyperparameters": current_hyperparams,
                "function_value": float(tune_res.fun),  # convert numpy float to python float if needed
                "x_values": [i.item() if isinstance(i, np.generic) else i for i in tune_res.x], # convert numpy array to python list
                "success": bool(tune_res.success),
                "func_calls": int(tune_res.nfev),       # convert numpy int to python int if needed
                "duration": time.time() - t0
            })

            if tune_res.fun < best_fun_value:
                best_fun_value = tune_res.fun
                best_hyperparams = current_hyperparams
                best_result = tune_res

            logger.info(f"Hyperparameters: {current_hyperparams}")
            logger.info(f"Func value: {tune_res.fun}")
            logger.info(f"Time (s): {time.time() - t0}\n")



        logger.info(f"Best function value: {best_fun_value}")
        logger.info(f"Best hyperparameters: {best_hyperparams}\n")

        with open(os.path.join(output_dir, f'tuning_results.yaml'), 'w') as file:
            yaml.dump(tuning_results, file, sort_keys=False)


    # start MLE optimization runs
    for i in tqdm(range(n_trials)):
        logger.info(f"starting optimization run {i}")
        t0 = time.time()
        initial_guess = uf.get_p0(p_bounds,1)[0]
        logger.info(f"initial guess: {initial_guess}")
       
        if mle_method == "differential_evolution":
            if tuning:
                res = sp.optimize.differential_evolution(optimization_func, p_bounds, args=optimization_func_args,  **best_hyperparams)
            else:
                res = sp.optimize.differential_evolution(optimization_func, p_bounds, args=optimization_func_args)
        elif mle_method == "basinhopping":
            if tuning:
                minimizer_kwargs = {"args": (rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, simulation_kwargs, p_lb, p_ub), "bounds": p_bounds}
                res = sp.optimize.basinhopping(optimization_func, initial_guess, minimizer_kwargs=minimizer_kwargs, **best_hyperparams)
            else:
                minimizer_kwargs = {"args": (rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, simulation_kwargs, p_lb, p_ub), "bounds": p_bounds}
                res = sp.optimize.basinhopping(optimization_func, initial_guess, minimizer_kwargs=minimizer_kwargs)
        elif mle_method == "dual_annealing":
            if tuning:
                res = sp.optimize.dual_annealing(optimization_func, p_bounds, args=optimization_func_args,  **best_hyperparams)
            else:
                res = sp.optimize.dual_annealing(optimization_func, p_bounds, args=optimization_func_args)
        elif mle_method == "shgo":
            if tuning:
                res = sp.optimize.shgo(optimization_func, p_bounds, args=optimization_func_args,  **best_hyperparams)
            else:
                res = sp.optimize.shgo(optimization_func, p_bounds, args=optimization_func_args)
        elif mle_method == "Nelder-Mead":
            if tuning:
                res = sp.optimize.minimize(optimization_func, initial_guess, args=(rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, simulation_kwargs, p_lb, p_ub), bounds=p_bounds, method="Nelder-Mead",  options=best_hyperparams)
            else:
                res = sp.optimize.minimize(optimization_func, initial_guess, args=(rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, simulation_kwargs, p_lb, p_ub), bounds=p_bounds, method="Nelder-Mead")
        elif mle_method == "Powell":
            if tuning:
                res = sp.optimize.minimize(optimization_func, initial_guess, args=(rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, simulation_kwargs, p_lb, p_ub),  bounds=p_bounds, method="Powell",  options=best_hyperparams)
            else:
                res = sp.optimize.minimize(optimization_func, initial_guess, args=(rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, simulation_kwargs, p_lb, p_ub),  bounds=p_bounds, method="Powell")
        elif mle_method == "CG":
            if tuning:
                res = sp.optimize.minimize(optimization_func, initial_guess, args=(rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, simulation_kwargs, p_lb, p_ub),  method="CG",  options=best_hyperparams)
            else:
                res = sp.optimize.minimize(optimization_func, initial_guess, args=(rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, simulation_kwargs, p_lb, p_ub),  method="CG", ) 
        elif mle_method == "L-BFGS-B":
            if tuning:
                res = sp.optimize.minimize(optimization_func, initial_guess, args=(rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, simulation_kwargs, p_lb, p_ub),  bounds=p_bounds, method="L-BFGS-B",   options=best_hyperparams) 
            else:
                res = sp.optimize.minimize(optimization_func, initial_guess, args=(rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, simulation_kwargs, p_lb, p_ub),  bounds=p_bounds, method="L-BFGS-B") 
        elif mle_method == "COBYLA":
            if tuning:
                res = sp.optimize.minimize(optimization_func, initial_guess, args=(rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, simulation_kwargs, p_lb, p_ub),  method="COBYLA",  options=best_hyperparams)
            else:
                res = sp.optimize.minimize(optimization_func, initial_guess, args=(rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, simulation_kwargs, p_lb, p_ub),  method="COBYLA")
        elif mle_method == "direct":
            if tuning:
                res = sp.optimize.direct(optimization_func, p_bounds, args=(rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, simulation_kwargs, p_lb, p_ub), **best_hyperparams)
            else:
                res = sp.optimize.direct(optimization_func, p_bounds, args=(rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, simulation_kwargs, p_lb, p_ub))
        elif mle_method == "SLSQP":
            if tuning:
                res = sp.optimize.minimize(optimization_func, initial_guess, args=(rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, simulation_kwargs, p_lb, p_ub),  method="SLSQP", options=best_hyperparams)
            else:    
                res = sp.optimize.minimize(optimization_func, initial_guess, args=(rr_model, y_obs, initial_conditions, initial_conditions_scale, buffer_concentration_scale, simulation_kwargs, p_lb, p_ub),  method="SLSQP")
        else:
            print('invalid MLE method.')
            assert(1==0)


        dt = time.time() - t0
        logger.info(f"method: {mle_method}")
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

    # Convert each NumPy array in results_list to a standard Python list
    results_dict = {
        'run': output_fname,
        'method': mle_method,
        'x': [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in results_list[0]],
        'fun': [float(i) for i in results_list[1]],
        'success': [bool(i) for i in results_list[2]],
        'nfev': results_list[3],
        't': results_list[4]
    }

    with open(os.path.join(output_dir, f'optimization_results.yaml'), 'w') as file:
        yaml.dump(results_dict, file, sort_keys=False)


if __name__ == '__main__':

    ##### Adjust this if needed ##### 
    example_config = "/example/antiporter_1_1_12D_cycle1_config.yaml"
    run_optimizer(example_config)