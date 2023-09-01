import roadrunner
import numpy as np


def simulate_assay(rr, rate_constants, initial_conditions, initial_conditions_scale, buffer_concentration_scale, solver_arguments):
    """
    Simulate the SSME assay using the provided rate constants, initial conditions, and solver arguments.

    Args:
        rr (object): RoadRunner object for simulating biochemical network models (e.g. transporter cycle).
        rate_constants (list): List of rate constant values. Must be in the same order as rate constant names in solver arguments.
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
        results (ndarray): Array of simulation results.
    """
    # Load simulation settings
    roadrunner_solver_output_selections = solver_arguments['roadrunner_solver_output_selections']
    buffer_concentration_sequence = solver_arguments['buffer_concentration_sequence']
    time_per_stage = solver_arguments['time_per_stage']
    n_points_per_stage = solver_arguments['n_points_per_stage']
    buffer_species_names = solver_arguments['buffer_species_names']
    absolute_tolerance = solver_arguments['solver_absolute_tolerance']
    relative_tolerance = solver_arguments['solver_relative_tolerance']
    remove_first_n = solver_arguments['remove_first_n']
    remove_after_n = solver_arguments['remove_after_n']
    species_labels = solver_arguments['species_labels']
    rate_constant_names = solver_arguments['rate_constant_names']

    # Reset roadrunner model and set integrator tolerances
    rr.resetToOrigin()
    rr.integrator.absolute_tolerance = absolute_tolerance
    rr.integrator.relative_tolerance = relative_tolerance

    # Update model rate constants
    parameters = dict(zip(rate_constant_names, rate_constants))
    for name, value in parameters.items():
        rr[name] = value

    # Run synthetic SSME assay
    ## iterate through buffer concentration sequence for assay
    results = []
    for i, solution in enumerate(buffer_concentration_sequence):  # Update buffer solution for each assay stage
        for j, label in enumerate(buffer_species_names):
            buffer_concentration_j = solution[j]*buffer_concentration_scale[j]  # scale buffer concentrations (obs = pred*scale_factor)
            setattr(rr, label, buffer_concentration_j)
        if i==0:  # Set initial state concentrations for stage 1 (equilibration)
            for j, label in enumerate(species_labels):
                initial_concentration_j = initial_conditions[j]*initial_conditions_scale[j]
                setattr(rr, label, initial_concentration_j)
            rr.simulate(i,i+time_per_stage,n_points_per_stage, selections=roadrunner_solver_output_selections)  # Don't store equilibration results
        else:
            tmp = rr.simulate(i,i+time_per_stage,n_points_per_stage, selections=roadrunner_solver_output_selections)
            results.append(tmp[remove_first_n:remove_after_n])
    return np.vstack(results).T

