import pytest
import roadrunner
import numpy as np
from ssme_function import simulate_assay  # Replace 'your_script_name' with the name of your script


# Load the model into roadrunner
rr = roadrunner.RoadRunner('./example/antiporter_1_1_cycle1_0v_SBML.xml')

# Test parameters 
rate_constants = [
    1e10,  # log10_k1_f
    1e3,   # log10_k1_r
    1e2,   # log10_k2_f
    1e2,   # log10_k2_r
    1e7,   # log10_k3_f
    1e3,   # log10_k3_r
    1e3,   # log10_k4_f
    1e10,  # log10_k4_r
    1e2,   # log10_k5_f
    1e2,   # log10_k5_r
    1e3    # log10_k6_f
]
initial_conditions = [0.0004369941792375325, 0, 0, 0, 0, 0, 1.e-7, 1.e-3]
initial_conditions_scale = [1, 1, 1, 1, 1, 1, 1, 1]
buffer_concentration_scale = [1, 1]
solver_arguments = {
    'buffer_species_names': ["conc_H_out", "conc_S_out"],
    'buffer_concentration_sequence': [[1.e-7,1.e-3],[0.5e-7,1.0e-3],[1.e-7,1.e-3]],
    'buffer_concentration_scale': [1,1],
    'time_per_stage': 1,
    'n_points_per_stage': 200,
    'remove_first_n': 1,
    'remove_after_n': 100,
    'rate_constant_names': ["k1_f_0", "k1_r_0", "k2_f_0", "k2_r_0", "k3_f_0", "k3_r_0", "k4_f_0", "k4_r_0", "k5_f_0", "k5_r_0", "k6_f_0"],
    'species_labels': ["conc_OF", "conc_OF_Hb", "conc_IF_Hb", "conc_IF_Hb_Sb", "conc_IF_Sb", "conc_OF_Sb", "conc_H_in", "conc_S_in"],
    'solver_absolute_tolerance': 1.e-15,
    'solver_relative_tolerance': 1.e-12,
    'roadrunner_solver_output_selections': ["time","I_t_total"]
}

# reference dataset
ref_data = np.loadtxt('./example/test_synthetic_data.csv', delimiter=',')

def test_simulate_assay():
    result = simulate_assay(rr, rate_constants, initial_conditions, initial_conditions_scale, buffer_concentration_scale, solver_arguments)
    assert result is not None, "Simulation result is None"
    assert result.shape == (2, 198), f"Expected shape (2, 198), but got {result.shape}"  # Assuming 2 stages with 99 points each (after removing 1) and 2 columns ("time" and "I_t_total")
    assert np.array_equal(result, ref_data), "Simulation result does not match the reference data"



# To run the test directly from this script
if __name__ == "__main__":
    test_simulate_assay()