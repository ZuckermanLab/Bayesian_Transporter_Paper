import amici
import time
import numpy as np
import matplotlib.pyplot as plt


sbmlfile = '/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/scratch/antiporter_12D_model_3c_no_events.xml'
# amici tests
sbml_importer = amici.SbmlImporter(sbmlfile)
model_name = 'model_transporter'
model_dir = 'model_dir'
sbml_importer.sbml2amici(model_name, model_dir)
# load the model module
model_module = amici.import_model_module(model_name, model_dir)
# instantiate model
model = model_module.getModel()
# instantiate solver
solver = model.getSolver()

print(model.getStateNames())
print(model.getParameterNames())


n_points = 50


H_out = [1e-7]
S_out = [1e-3]
trasporter_states = [0.0011694210430300167]

buffer_dict = {
    'H_out':H_out[0],
    'S_out':S_out[0],
}

y0_dict = {
    'H_out':H_out[0],
    'S_out':S_out[0],
    'H_in':1e-7,
    'S_in':1e-3,
    'OF':0.0011694210430300167,
    'OF_Hb':0,
    'IF_Hb':0,
    'IF_Hb_Sb':0,
    'IF_Sb':0,
    'OF_Sb':0,
}

# ('OF', 'OF_Hb', 'IF_Hb', 'IF_Hb_Sb', 'IF_Sb', 'OF_Sb', 'H_in', 'S_in', 'H_out', 'S_out')
x0 = [0.0011694210430300167, 0, 0, 0, 0, 0, 1e-7, 1e-3, 1-7, 1e-3]

solver.setRelativeTolerance(1e-8)
solver.setAbsoluteTolerance(1e-12)
solver.setMaxSteps(1000 * solver.getMaxSteps())
model.setInitialStates(x0)

model.setTimepoints(np.linspace(0,1,n_points))
rdata = amici.runAmiciSimulation(model, solver)
print(rdata)