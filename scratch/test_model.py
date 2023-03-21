import unittest
import tellurium as te

class TestAntiporterModel(unittest.TestCase):

    def setUp(self):
        ant_model_file = '/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/scratch/antiporter_12D_model_3c.txt'
        self.ant_model_file = ant_model_file
        self.m = te.loada(ant_model_file)

    def test_simulate_model(self):
        selections = [
            'time', 
            'rxn1', 
            'rxn2', 
            'rxn3', 
            'rxn4', 
            'rxn5', 
            'rxn6', 
            'current', 
            'OF', 
            'OF_Hb',
            'IF_Hb',
            'IF_Hb_Sb', 
            'IF_Sb', 
            'OF_Sb',
            'H_out',
            'H_in', 
            'S_out',
            'S_in', 
            'transporter_amount', 
            'H_conc_ratio', 
            'S_conc_ratio'
        ]
        self.m.resetToOrigin()
        self.m.H_act = 5e-7  # plot values
        self.m.integrator.absolute_tolerance = 1e-22
        self.m.integrator.relative_tolerance = 1e-12
        results = self.m.simulate(0,3,180, selections=selections)
        

if __name__ == '__main__':
    unittest.main()