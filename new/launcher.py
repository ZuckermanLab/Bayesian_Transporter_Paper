import numpy as np
import yaml
import tellurium as te
import os
import shutil
import datetime
import logging
import matplotlib.pyplot as plt
import importlib
import json
import pandas as pd
import argparse

import ssme_simulator 
from run_optimizer import run_optimization
#import run_sampler


def create_output_directories(sbml_model_name, experiment_name, optimization_and_inference_name, random_seed, sbml_model_file, observed_data_file, config_file, current_file):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
    output_fname = f'run_{sbml_model_name}_{experiment_name}_{optimization_and_inference_name}_r{random_seed}'
    output_dir = output_fname + '_' + timestamp
    os.mkdir(output_dir)
    misc_dir = os.path.join(output_dir, "misc")
    os.mkdir(misc_dir)
    shutil.copy(sbml_model_file, misc_dir)
    shutil.copy(observed_data_file, misc_dir)
    shutil.copy(config_file, misc_dir)
    shutil.copy(current_file, misc_dir)
    return output_dir, misc_dir


def setup_logger(output_dir, random_seed, log_file_name='log_file.log'):
    logger = logging.getLogger('log')
    logger.setLevel(logging.INFO) 
    file_handler = logging.FileHandler(os.path.join(output_dir, log_file_name))
    file_handler.setLevel(logging.INFO)
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    logger.info(f"output_dir: {output_dir}")
    logger.info(f"seed: {random_seed}")
    return logger


def get_parameter_bounds_and_nominals(parameter_settings):
    parameter_names = [param["name"] for param in parameter_settings]
    parameter_lower_bounds = [param["bounds"][0] for param in parameter_settings]
    parameter_upper_bounds = [param["bounds"][1] for param in parameter_settings]
    parameter_bounds = list(zip(parameter_lower_bounds,parameter_upper_bounds))
    parameter_nominals = [param["nominal"] for param in parameter_settings]
    return parameter_names, parameter_bounds, parameter_nominals


def get_y_nom(parameter_nominals, solver_arguments, ssme_experiment, rr_model):
    sbml_model_parameters_nominal_log10 = parameter_nominals[:len(solver_arguments['sbml_model_parameter_names'])]
    sbml_model_parameters_nominal = [10**p for p in sbml_model_parameters_nominal_log10]
    sigma_nominal_log10 = parameter_nominals[-1]
    sigma_nominal = 10**sigma_nominal_log10
    results_nom = ssme_experiment.simulate_assay(sbml_model_parameters_nominal, rr_model)
    y_nom = results_nom[1]
    return y_nom


def plot_save_data(y_nom, y_obs, output_dir):
    # Plot data
    plt.figure(figsize=(8,6))
    plt.title('Net ion influx (M/s) vs simulation step')
    plt.plot(y_nom, label='y_nom')
    plt.plot(y_obs, 'o', label='y_obs', alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'net_ion_influx_trace_nom_and_obs.png'))


def load_config_file(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Get general settings
    general_settings = config["general_configuration"]
    sbml_model_file = general_settings["sbml_model_file"]
    observed_data_file = general_settings["observed_data_file"]
    sbml_model_name = general_settings["sbml_model_name"]
    random_seed = general_settings["random_seed"]

    # Get extended_solver settings from the YAML configuration
    initial_conditions = config["experiment_simulation_configuration"]["species_initial_concentrations"]
    solver_arguments = config["experiment_simulation_configuration"]
    experiment_name = solver_arguments["experiment_name"]

    # Get optimization and MCMC settings
    use_optimization = config["optimization_and_inference"]["use_optimization"]
    use_mcmc = config["optimization_and_inference"]["use_mcmc"]
    optimization_settings = config["optimization_and_inference"]["optimization_arguments"]
    mcmc_settings = config["optimization_and_inference"]["mcmc_arguments"]
    parameter_settings = config["optimization_and_inference"]["parameters"]
    optimization_and_inference_name = config["optimization_and_inference"]["optimization_and_inference_name"]
    
    return (config, general_settings, sbml_model_file, observed_data_file, sbml_model_name, random_seed, initial_conditions,
           solver_arguments, experiment_name, use_optimization, use_mcmc, optimization_settings, mcmc_settings,
           parameter_settings, optimization_and_inference_name)



if __name__ == '__main__':


    ##### INITIALIZATION #####
    # open and load configuration file
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Bayesian_Transporter Optimization")

    # Add an argument for the config file
    parser.add_argument(
        "-c",
        "--config",
        dest="config_file",
        required=True,
        help="Path to the configuration file",
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Get the config file path from the parsed arguments
    config_file = args.config_file

    config, general_settings, sbml_model_file, observed_data_file, sbml_model_name, random_seed, initial_conditions, \
    solver_arguments, experiment_name, use_optimization, use_mcmc, optimization_settings, mcmc_settings, \
    parameter_settings, optimization_and_inference_name = load_config_file(config_file)

    # create new directory for runs and sub directory for misc files
    output_dir, misc_dir = create_output_directories(sbml_model_name, 
                                                     experiment_name, 
                                                     optimization_and_inference_name, 
                                                     random_seed, sbml_model_file, 
                                                     observed_data_file, 
                                                     config_file, 
                                                     os.path.abspath(__file__))

    logger = setup_logger(output_dir, random_seed)   # create logging file 
    ssme_experiment = ssme_simulator.ExtendedSolver(initial_conditions, solver_arguments)  # Create an instance of the ExtendedSolver class
    parameter_names, parameter_bounds, parameter_nominals = get_parameter_bounds_and_nominals(parameter_settings)  # Get parameter bounds and nominal values
    rr_model = te.loadSBMLModel(sbml_model_file)   # load SBML model as a roadrunner object
    y_obs = np.genfromtxt(observed_data_file, delimiter=',')  # load observed data file
    y_nom = get_y_nom(parameter_nominals, solver_arguments, ssme_experiment, rr_model)   # get predicted data using nominal values
    
    np.savetxt("y_nom_model1_exp1and2.csv", y_nom)
    plt.plot(y_nom)
    plt.savefig('y_nom.png')
    plot_save_data(y_nom, y_obs, misc_dir)   # save observed vs nominal data plot 

    assert(1==0)

    ##### RUN OPTIMIZER AND/OR MCMC SAMPLER #####
    
    if use_optimization:
        objective_function_module_name = config["optimization_and_inference"]["optimization_arguments"]["objective_function_module"]
        objective_function_name = config["optimization_and_inference"]["optimization_arguments"]["objective_function_name"]
        objective_function_module = importlib.import_module(objective_function_module_name)
        obj_func = getattr(objective_function_module, objective_function_name)
        opt_method= config["optimization_and_inference"]["optimization_arguments"]["method"]
        opt_method_args = None
        n_replicas_optimize = optimization_settings['n_replicas']

        # model specific
        use_extended = config["optimization_and_inference"]["use_extended"]
        logger.info(f"running optimization")
        optimization_dir = os.path.join(output_dir, "optimization")
        os.mkdir(optimization_dir)

        # Pass the arguments as a tuple
        lower_bounds = [bound[0] for bound in parameter_bounds]
        upper_bounds = [bound[1] for bound in parameter_bounds]

        obj_func_args = (y_obs, rr_model, ssme_experiment, lower_bounds, upper_bounds, use_extended)

        result_dict = {}
        for method in opt_method:
            # Call the run_replicas function with the tuple of arguments
            results = run_optimization(
                objective_function=obj_func,
                obj_func_args=obj_func_args,
                method = method, 
                bounds = parameter_bounds, 
                n_replicas = n_replicas_optimize, 
                method_args = {}, 
                random_seed= random_seed,
                output_directory = optimization_dir, 
                initial_values=None, 
                logger=logger
            )
            logger.info(f"finished optimization using {method} method")
            result_dict[method] = results
        logger.info(f"finished optimization")

        combined_df = pd.DataFrame()   
        for method, results in result_dict.items():
            df = pd.DataFrame(results)
            df['method'] = method
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        csv_file = os.path.join(optimization_dir, "combined_results_all.csv")
        combined_df.to_csv(csv_file, index=False)



    if use_mcmc:
        logger.info(f"running mcmc")
        mcmc_dir = os.path.join(output_dir, "mcmc")
        os.mkdir(mcmc_dir)

        objective_function_module_name = config["optimization_and_inference"]["optimization_arguments"]["objective_function_module"]
        objective_function_name = config["optimization_and_inference"]["optimization_arguments"]["objective_function_name"]
        objective_function_module = importlib.import_module(objective_function_module_name)
        obj_func = getattr(objective_function_module, objective_function_name)
        opt_method= config["optimization_and_inference"]["optimization_arguments"]["method"]
        opt_method_args = None
        n_replicas_optimize = optimization_settings['n_replicas']

        # model specific
        use_extended = config["optimization_and_inference"]["use_extended"]
        logger.info(f"running optimization")
        optimization_dir = os.path.join(output_dir, "optimization")
        os.mkdir(optimization_dir)

        # Pass the arguments as a tuple
        lower_bounds = [bound[0] for bound in parameter_bounds]
        upper_bounds = [bound[1] for bound in parameter_bounds]

        obj_func_args = (y_obs, rr_model, ssme_experiment, lower_bounds, upper_bounds, use_extended)

        result_dict = {}
        for method in opt_method:
            # Call the run_replicas function with the tuple of arguments
            results = run_optimization(
                objective_function=obj_func,
                obj_func_args=obj_func_args,
                method = method, 
                bounds = parameter_bounds, 
                n_replicas = n_replicas_optimize, 
                method_args = {}, 
                random_seed= random_seed,
                output_directory = optimization_dir, 
                initial_values=None, 
                logger=logger
            )
            logger.info(f"finished optimization using {method} method")
            result_dict[method] = results
        logger.info(f"finished optimization")

        combined_df = pd.DataFrame()   
        for method, results in result_dict.items():
            df = pd.DataFrame(results)
            df['method'] = method
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        csv_file = os.path.join(optimization_dir, "combined_results_all.csv")
        combined_df.to_csv(csv_file, index=False)

        # run_mcmc(args) for m replicas
        logger.info(f"finished mcmc")
