import roadrunner
import numpy as np


def simulate_assay(rr, rate_constants, initial_conditions, initial_conditions_scale, buffer_concentration_scale, solver_arguments):
    """
    Simulate the SSME assay.

    Args:
        rate_constants (list): List of rate constant values. 
        initial_conditions (list): List of initial species concentrations.
        initial_conditions_scale (list): List of scaling factors for the initial species concentrations.
        buffer_concentration_scale (list): List of scaling factors for the buffer concentrations.

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
    results = []
    for i, solution in enumerate(buffer_concentration_sequence):  # Update buffer solution for each assay stage
        for j, label in enumerate(buffer_species_names):
            buffer_concentration_j = solution[j]*buffer_concentration_scale[j]
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

