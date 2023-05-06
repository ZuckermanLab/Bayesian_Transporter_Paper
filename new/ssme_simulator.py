import numpy as np


class ExtendedSolver:
    def __init__(self, initial_conditions,  solver_arguments, initial_conditions_scale=None, buffer_concentration_scale=None):
        self.initial_conditions = initial_conditions
        self.solver_arguments = solver_arguments
        self.initial_conditions_scale = initial_conditions_scale if initial_conditions_scale is not None else [1] * len(initial_conditions)
        self.buffer_concentration_scale = buffer_concentration_scale if buffer_concentration_scale is not None else [1] * len(solver_arguments['buffer_species_names'])

        # Load simulation settings
        self.roadrunner_solver_output_selections = solver_arguments['roadrunner_solver_output_selections']
        self.buffer_concentration_sequence = solver_arguments['buffer_concentration_sequence']
        self.time_per_stage = solver_arguments['time_per_stage']
        self.n_points_per_stage = solver_arguments['n_points_per_stage']
        self.buffer_species_names = solver_arguments['buffer_species_names']
        self.absolute_tolerance = solver_arguments['solver_absolute_tolerance']
        self.relative_tolerance = solver_arguments['solver_relative_tolerance']
        self.remove_first_n = solver_arguments['remove_first_n']
        self.remove_after_n = solver_arguments['remove_after_n']
        self.species_labels = solver_arguments['species_labels']
        self.smbl_model_parameter_names = solver_arguments['sbml_model_parameter_names']


    def update_scales(self, initial_conditions_scale=None, buffer_concentration_scale=None):
        if initial_conditions_scale is not None:
            self.initial_conditions_scale = initial_conditions_scale
        if buffer_concentration_scale is not None:
            self.buffer_concentration_scale = buffer_concentration_scale


    def simulate_assay(self, sbml_model_parameters, rr):
        # Reset roadrunner model and set integrator tolerances
        rr.resetToOrigin()
        rr.integrator.absolute_tolerance = self.absolute_tolerance
        rr.integrator.relative_tolerance = self.relative_tolerance

        # Update model rate constants and other model parameters (i.e. capacitance)
        parameters = dict(zip(self.smbl_model_parameter_names, sbml_model_parameters))
        for name, value in parameters.items():
            rr[name] = value

        # Run synthetic SSME assay
        results = []
        for i, solution in enumerate(self.buffer_concentration_sequence):  # Update buffer solution for each assay stage
            for j, label in enumerate(self.buffer_species_names):
                buffer_concentration_j = solution[j]*self.buffer_concentration_scale[j]
                setattr(rr, label, buffer_concentration_j)

            if i==0:  # Set initial state concentrations for stage 1 (equilibration)
                for j, label in enumerate(self.species_labels):
                    initial_concentration_j = self.initial_conditions[j]*self.initial_conditions_scale[j]
                    setattr(rr, label, initial_concentration_j)
                rr.simulate(i,i+self.time_per_stage,self.n_points_per_stage, selections=self.roadrunner_solver_output_selections)  # Don't store equilibration results
            else:
                tmp = rr.simulate(i,i+self.time_per_stage,self.n_points_per_stage, selections=self.roadrunner_solver_output_selections)
                results.append(tmp[self.remove_first_n:self.remove_after_n])
        return np.vstack(results).T


if __name__ == '__main__':
    pass
