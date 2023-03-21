import sampler 
import ssme
import utility

import json
import numpy
import scipy
import matplotlib.pyplot as plt
import pandas as pd

import tellurium

import time


if __name__ == '__main__':
    config_fname = "/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/data/config_15D_real_data_v2.json"
    config = utility.parse_config_file(config_fname)

    model_file = config['model_file']
    data_file = config['data_file']
    simulation_kwargs = config['simulation_kwargs']
    model_parameters = config['model_parameters']
    run_kwargs = config['run_kwargs']
    utility.save_config_file(config, f"{run_kwargs['output_label']}_log.txt")
    
    seed = run_kwargs['seed']
    numpy.random.seed(seed)
    param_ref = [value[2] for key, value in model_parameters.items()]
    param_bounds = [[value[0], value[1]] for key, value in model_parameters.items()]
    k_ref = param_ref[:-1]
    sigma_ref = param_ref[-1]
 
    rr_model = utility.load_rr_model_from_sbml(model_file)
    res = ssme.simulate_y_pred_rr(rr_model,k_ref,**simulation_kwargs)
    y_ref = res[1]
    
    y_obs = numpy.loadtxt(data_file, delimiter=',')

    logl_ref = sampler.calc_normal_log_likelihood(y_obs,y_ref,10**sigma_ref)
 
    print(f'using {config_fname} configuration')
    print(f'log likelihood reference: {logl_ref}')


    def obj_func(params, rr_model, y_obs, sim_args):
        k = params[:-1]
        sigma = 10**params[-1]
        try:
            res = ssme.simulate_y_pred_rr(rr_model, k, **sim_args)
            y_pred = res[1]
            return -1*sampler.calc_normal_log_likelihood(y_obs, y_pred, sigma)
        except:
            return 1e30
    
    results_dict = {}

    # Define the optimization methods (no shgo or other constrained methods)
    #methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr',  'basinhopping', 'differential_evolution', 'dual_annealing', 'direct']

    methods = ['differential_evolution']
    N_iter = 10

    params_init_array = sampler.get_random_param_values(N_iter,param_bounds, method='LHS')
   
    # Run each optimization method and print the runtime
    for method in methods:
        print(f'{method}')
        success_list = []
        dt_list = []
        x_list = []
        y_list = []
        x_rmsd_list = []
        y_rmsd_list = []

        for i in range(N_iter):
            params_init = params_init_array[i]
            t0 = time.time()
            if method in ['Nelder-Mead', 'L-BFGS-B', 'TNC', 'SLSQP', 'Powell', 'trust-constr']:
                result = scipy.optimize.minimize(obj_func, x0=params_init, method=method, bounds=param_bounds, args=(rr_model, y_obs, simulation_kwargs), options={'maxiter':1e5})
            elif method == 'basinhopping':
                result = scipy.optimize.basinhopping(obj_func, x0=params_init, minimizer_kwargs={'args':(rr_model, y_obs, simulation_kwargs)})
            elif method == 'differential_evolution':
                result = scipy.optimize.differential_evolution(obj_func, bounds=param_bounds, args=(rr_model, y_obs, simulation_kwargs))
            elif method == 'shgo':
                result = scipy.optimize.shgo(obj_func, bounds=param_bounds, args=(rr_model, y_obs, simulation_kwargs))
            elif method == 'dual_annealing':
                result = scipy.optimize.dual_annealing(obj_func, bounds=param_bounds, args=(rr_model, y_obs, simulation_kwargs), x0=params_init)
            elif method == 'direct':
                result = scipy.optimize.direct(obj_func, bounds=param_bounds, args=(rr_model, y_obs, simulation_kwargs))
            else:
                result = scipy.optimize.minimize(obj_func, x0=params_init, method=method, args=(rr_model, y_obs, simulation_kwargs))
            tf = time.time()
            x_temp = [i for i in result.x]
            y_temp = -1*result.fun  # negative log-likelihood --> log-likelihood
            x_rmsd = numpy.sqrt(numpy.mean(numpy.square(numpy.array(param_ref)-numpy.array(x_temp))))
            y_rmsd = numpy.sqrt(numpy.mean(numpy.square(logl_ref-y_temp)))
            dt = tf-t0

            success_list.append(result.success)
            dt_list.append(dt)
            x_list.append(x_temp)
            y_list.append(y_temp)
            x_rmsd_list.append(x_rmsd)
            y_rmsd_list.append(y_rmsd)

        print(f'finished {N_iter} iterations of {method} MLE optimization')

        y_rmsd_mean = numpy.mean(y_rmsd_list)
        y_rmsd_std = numpy.std(y_rmsd_list)
        x_rmsd_mean = numpy.mean(x_rmsd_list)
        x_rmsd_std = numpy.std(x_rmsd_list)
        dt_mean = numpy.mean(dt_list)
        results_dict[method] = [success_list, dt_list, y_list, x_list, y_rmsd_list, x_rmsd_list, y_rmsd_mean, x_rmsd_mean, y_rmsd_std, x_rmsd_std, dt_mean]

    df = pd.DataFrame.from_dict(results_dict, orient='index',columns=['success', 'runtime_s','mle','mle_x','rmsd_mle', 'rmsd_x','rmsd_mle_mean', 'rmsd_x_mean','rmsd_mle_std', 'rmsd_x_std','dt_mean'])
    df.to_csv("MLE_comparison_results_15D_gdx_ssme_2_DE.csv")
    print(df)
