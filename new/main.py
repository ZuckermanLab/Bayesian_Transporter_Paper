from ssme_solver import SSMESolver
import yaml
import numpy as np
import matplotlib.pyplot as plt 
import emcee




config_file = "/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/new/antiporter_1_1_12D_cycle1/experiment1/antiporter_1_1_12D_cycle1_experiment1_config.yaml"

# Load the config.yaml file and iniitalize values
with open(config_file, "r") as f:
    config = yaml.safe_load(f)



k_names = config['solver_arguments']['rate_constant_names']
k_nom = [10**d['nominal'] for d in config['bayesian_inference']['parameters']]
parameters = k_nom#dict(zip(k_names,k_nom))

initial_conditions = config['solver_arguments']['species_initial_concentrations']
initial_conditions_scale = config['solver_arguments']['species_initial_concentrations_scale']
buffer_concentration_scale = config['solver_arguments']['buffer_concentration_scale']




# Create an SSMESolver instance and run the solver
solver = SSMESolver(config['model_file'], config['solver_arguments'])
solver.load_model()

data = solver.simulate(parameters, initial_conditions, initial_conditions_scale, buffer_concentration_scale)



print(np.shape(data))

t = data[0]
y = data[1]

plt.figure(figsize=(10,5))
plt.plot(y)
sigma_true = k_nom[-1]
y_obs = y + np.random.normal(0,sigma_true, np.size(y))
plt.plot(y_obs, 'o')
plt.savefig('test_new.png')
np.savetxt('dataset.csv', y_obs, delimiter=',')
# samples_fnames = [
#     "/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/samples_gdx_2_1_exp1_w_exp_params_r2.csv",
#     "/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/samples_gdx_2_1_exp1_w_exp_params_r4.csv",
#     "/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/samples_gdx_2_1_exp1_w_exp_params_r6.csv",
#     "/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/samples_gdx_2_1_exp1_w_exp_params_r8.csv",
# ]

# burn_in = 100000 
# samples_full_list = [np.loadtxt(f, delimiter=',') for f in samples_fnames]
# samples_list = [s[burn_in:, :] for s in samples_full_list]
# print(np.shape(samples_list))
# inf_data = az.convert_to_dataset(np.array(samples_list))
# print(inf_data)

# print('sampling diagnostics')
# print(f'ESS: {az.ess(inf_data)}')
# print(f'rhat: {az.rhat(inf_data)}')
# print(f'mcse: {az.mcse(inf_data)}')

# print(az.summary(inf_data))

