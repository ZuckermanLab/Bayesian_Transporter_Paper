import numpy as np
import os
import json
from scipy.optimize import minimize
from scipy.optimize import LbfgsInvHessProduct
import scipy
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import csv
from scipy.optimize import basinhopping, brute, differential_evolution, shgo, dual_annealing, direct


def run_optimization(
    objective_function,
    obj_func_args,
    method,
    bounds,
    n_replicas,
    method_args={},
    random_seed=0,
    output_directory="output",
    initial_values=None,
    logger=None, 
):
    # Initialize the random seed
    np.random.seed(random_seed)

    method_output_directory = os.path.join(output_directory, method)
    if not os.path.exists(method_output_directory):
        os.makedirs(method_output_directory)

    # Run the optimization replicas
    results = []
    output_results = []

    for replica in tqdm(range(n_replicas)):
        result = run_single_optimization(
            replica,
            objective_function,
            obj_func_args,
            method,
            method_args,
            bounds,
            random_seed,
            initial_values,
        )
        results.append(result)

        output_results.append([int(random_seed + replica), result.fun, result.nfev, result['time']] + result.x.tolist() + [method])
        logger.info(f"optimization replica run finished.\n  replica={int(random_seed + replica)}\n  func={result.fun}\n  nfev={result.nfev}\n  time={result['time']}\n  x={result.x.tolist()}")
    with open(os.path.join(method_output_directory, f"{method}_all_replica_results.csv"), "w", newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["replica", "function_value", "nfev", "time"] + [f"x{i}" for i in range(len(bounds))] + ["method"])
        csv_writer.writerows(output_results)
    create_plots(np.array(output_results)[:, :-1].astype(float) , method_output_directory, len(bounds), method)
    return results



def run_single_optimization(
    replica,
    objective_function,
    obj_func_args,
    method,
    method_args,
    bounds,
    random_seed=0,
    initial_values=None,
):
    # Set the random seed for this replica
    np.random.seed(random_seed + replica)

    # Generate initial values for the optimization if not provided
    if initial_values is None:
        x0 = generate_initial_values(bounds)
    else:
        x0 = initial_values[replica]

    # Run the optimization and measure runtime
    start_time = time.time()

    if method == "basinhopping":
        result = basinhopping(
            objective_function,
            x0,
            minimizer_kwargs={"args": obj_func_args, "bounds": bounds, **method_args},
        )
    elif method == "brute":
        result = brute(
            objective_function,
            ranges=bounds,
            args=obj_func_args,
            full_output=True,
            **method_args,
        )
    elif method == "differential_evolution":
        result = differential_evolution(
            objective_function,
            bounds,
            args=obj_func_args,
            **method_args,
        )
    elif method == "shgo":
        result = shgo(
            objective_function,
            bounds,
            args=obj_func_args,
            **method_args,
        )
    elif method == "dual_annealing":
        result = dual_annealing(
            objective_function,
            bounds,
            args=obj_func_args,
            **method_args,
        )
    elif method == "direct":
        result = direct(
            objective_function,
            bounds,
            args=obj_func_args,
            **method_args,
        )
    else:  # Default to minimize
        result = minimize(
            objective_function,
            x0,
            args=obj_func_args,
            method=method,
            bounds=bounds,
            **method_args,
        )

    runtime = time.time() - start_time

    # Add runtime to the result dictionary
    result['time'] = runtime
    return result


def generate_initial_values(bounds):
    return [np.random.uniform(low, high) for low, high in bounds]


def create_plots(output_results, output_directory, n_params, method):
    output_results = np.array(output_results)

    # Create scatter plot of function values vs replica index
    plt.figure()
    plt.scatter(output_results[:, 0], output_results[:, 1])
    plt.xlabel("Replica Index")
    plt.ylabel("Function Value")
    plt.title(f"Function Value vs Replica Index using {method}")
    plt.savefig(os.path.join(output_directory, f"scatter_plot_function_values_{method}.png"))
    plt.close()

    # Create density histogram for function values
    plt.figure()
    plt.hist(output_results[:, 1], bins=30, density=True)
    plt.xlabel("Function Value")
    plt.ylabel("Density")
    plt.title(f"Density Histogram of Function Values using {method}")
    plt.savefig(os.path.join(output_directory, f"density_histogram_function_values_{method}.png"))
    plt.close()

    # Create density histograms for each x value
    grid_size = int(np.ceil(np.sqrt(n_params)))
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(3 * grid_size, 3 * grid_size))

    for i in range(n_params):
        row, col = i // grid_size, i % grid_size
        axs[row, col].hist(output_results[:, i + 4], bins=30, density=True)  # updated index to account for nfev and time columns
        axs[row, col].set_xlabel(f"x{i}")
        axs[row, col].set_ylabel("Density")
        axs[row, col].set_title(f"Distribution of x{i}")

    # Remove unused subplots
    for i in range(n_params, grid_size * grid_size):
        row, col = i // grid_size, i % grid_size
        fig.delaxes(axs[row, col])

    fig.tight_layout()
    plt.savefig(os.path.join(output_directory, f"density_histograms_x_values_{method}.png"))
    plt.close()

    # Create box and whisker plot with scatter overlay for each x value
    fig, ax = plt.subplots(figsize=(8, 6))
    box_data = [output_results[:, i + 4] for i in range(n_params)]  # updated index to account for nfev and time columns
    ax.boxplot(box_data, widths=0.5)

    for i, data in enumerate(box_data):
        x = np.random.normal(i + 1, 0.04, size=len(data))
        ax.scatter(x, data, alpha=0.5, s=8)

    ax.set_xticklabels([f"x{i}" for i in range(n_params)])
    ax.set_xlabel("Parameter Index")
    ax.set_ylabel("Value")
    ax.set_title(f"Box and Whisker Plot with Scatter Overlay for x Values using {method}")
    plt.savefig(os.path.join(output_directory, f"box_and_whisker_scatter_x_values_{method}.png"))
    plt.close()

    # Create multiplot of number of function evaluations and runtimes
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].hist(output_results[:, 2], bins=30, density=True)
    axs[0].set_xlabel("Number of Function Evaluations")
    axs[0].set_ylabel("Density")
    axs[0].set_title("Density Histogram of Number of Function Evaluations")

    axs[1].hist(output_results[:, 3], bins=30, density=True)
    axs[1].set_xlabel("Runtime")
    axs[1].set_ylabel("Density")
    axs[1].set_title("Density Histogram of Runtimes")

    fig.tight_layout()
    plt.savefig(os.path.join(output_directory, f"density_histograms_nfev_and_time_{method}.png"))
    plt.close()