import roadrunner
import numpy as np


class SSMESolver:
    """
    A class for simulating a synthetic single-molecule experiment (SSME).

    Attributes:
        model_file (str): Path to SBML file defining the model to simulate.
        solver_arguments (dict): Dictionary containing simulation settings.

    Methods:
        load_model(): Load the SBML model defined by model_file using roadrunner.
        set_rate_constants(rate_constants): Set the rate_constants values for the model.
        simulate(rate_constants, initial_conditions, initial_conditions_scale, buffer_concentration_scale): Simulate the SSME assay.
        run(rate_constants, initial_conditions, initial_conditions_scale, buffer_concentration_scale): Load the model and simulate the SSME assay.
    """

    def __init__(self, model_file, solver_arguments):
        """
        Initialize the SSMESolver object with a path to the SBML model file and simulation settings.

        Args:
            model_file (str): Path to SBML file defining the model to simulate.
            solver_arguments (dict): Dictionary containing simulation settings.
        """
        self.model_file = model_file
        self.solver_arguments = solver_arguments
        self.rate_constant_names = self.solver_arguments['rate_constant_names']


    def load_model(self):
        """
        Load the SBML model defined by self.model_file using roadrunner.
        """
        self.rr = roadrunner.RoadRunner(self.model_file)


    def set_rate_constants(self, rate_constants):
        """
        Set the parameter values for the model.

        Args:
            rate_constants (list): List of rate constant values. Must be same order as 'self.rate_constant_names'
        """

        parameters = dict(zip(self.rate_constant_names, rate_constants))
        for name, value in parameters.items():
            self.rr[name] = value


    def simulate(self, rate_constants, initial_conditions, initial_conditions_scale, buffer_concentration_scale):
        """
        Simulate the SSME assay.

        Args:
            rate_constants (list): List of rate constant values. Must be same order as 'self.rate_constant_names'
            initial_conditions (list): List of initial species concentrations.
            initial_conditions_scale (list): List of scaling factors for the initial species concentrations.
            buffer_concentration_scale (list): List of scaling factors for the buffer concentrations.

        Returns:
            results (ndarray): Array of simulation results.
        """
        # Load simulation settings
        roadrunner_solver_output_selections = self.solver_arguments['roadrunner_solver_output_selections']
        buffer_concentration_sequence = self.solver_arguments['buffer_concentration_sequence']
        time_per_stage = self.solver_arguments['time_per_stage']
        n_points_per_stage = self.solver_arguments['n_points_per_stage']
        buffer_species_names = self.solver_arguments['buffer_species_names']
        absolute_tolerance = self.solver_arguments['solver_absolute_tolerance']
        relative_tolerance = self.solver_arguments['solver_relative_tolerance']
        remove_first_n = self.solver_arguments['remove_first_n']
        remove_after_n = self.solver_arguments['remove_after_n']
        species_labels = self.solver_arguments['species_labels']

        # Reset roadrunner model and set integrator tolerances
        self.rr.resetToOrigin()
        self.rr.integrator.absolute_tolerance = absolute_tolerance
        self.rr.integrator.relative_tolerance = relative_tolerance

        # Update model rate constants
        self.set_rate_constants(rate_constants)

        # Run synthetic SSME assay
        results = []
        for i, solution in enumerate(buffer_concentration_sequence):  # Update buffer solution for each assay stage
            for j, label in enumerate(buffer_species_names):
                buffer_concentration_j = solution[j]*buffer_concentration_scale[j]
                setattr(self.rr, label, buffer_concentration_j)

            if i==0:  # Set initial state concentrations for stage 1 (equilibration)
                for j, label in enumerate(species_labels):
                    initial_concentration_j = initial_conditions[j]*initial_conditions_scale[j]
                    setattr(self.rr, label, initial_concentration_j)

                self.rr.simulate(i,i+time_per_stage,n_points_per_stage, selections=roadrunner_solver_output_selections)  # Don't store equilibration results
            else:
                tmp = self.rr.simulate(i,i+time_per_stage,n_points_per_stage, selections=roadrunner_solver_output_selections)
                results.append(tmp[remove_first_n:remove_after_n])
        return np.vstack(results).T


    def run(self, rate_constants, initial_conditions, initial_conditions_scale, buffer_concentration_scale):
        """
        Load the SBML model and simulate the SSME assay.

        Args:
            rate_constants (list): List of rate constant values.Must be same order as 'self.rate_constant_names'
            initial_conditions (list): List of initial species concentrations.
            initial_conditions_scale (list): List of scaling factors for the initial species concentrations.
            buffer_concentration_scale (list): List of scaling factors for the buffer concentrations.

        Returns:
            data (ndarray): Array of simulation results.
        """
        self.load_model()
        data = self.simulate(rate_constants, initial_conditions, initial_conditions_scale, buffer_concentration_scale)
        return data

    
