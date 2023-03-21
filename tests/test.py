import unittest
import roadrunner
import tellurium
import numpy
import json

import sys
sys.path.insert(0,'/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/scripts')

import utility
import ssme
import sampler


class TestUtility(unittest.TestCase):
    def test_convert(self):
        '''Unit test to check if a model file in Antimony format can be converted to SBML format'''
        # Input file in Antimony format
        in_fname = '/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/archive/v1/antiporter_12D_model.txt'
        # Output file in SBML format
        out_fname = 'test.xml'
        # Convert the input file to SBML format
        r = utility.convert_antimony_to_sbml(in_fname, out_fname)


    def test_load_rr_model(self):
        '''Unit test to check if an SBML file can be loaded as a roadrunner model'''
        # SBML model file
        fname = '/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/scratch/antiporter_12D_model_3c_no_events.xml'
        # Load the SBML file as a roadrunner model
        r = utility.load_rr_model_from_sbml(fname)
        # Assert if the output is a roadrunner model
        assert(type(r)==roadrunner.RoadRunner)


    def test_rr_model_to_ODE_type(self):
        '''Unit test to check if the ODEs of a roadrunner model can be obtained as a string'''
        # SBML model file
        fname = '/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/scratch/antiporter_12D_model_3c_no_events.xml'
        # Load the SBML file as a roadrunner model
        r = utility.load_rr_model_from_sbml(fname)
        # Obtain the ODEs of the roadrunner model as a string
        ODEs = utility.get_rr_model_ODEs(r)
        # Assert if the output is a string
        assert(type(ODEs)==str)


    def test_get_rr_model_ODEs(self):
        '''Unit test to check outputted roadrunner ODEs are correct'''
        # Define a simple model
        model_str = """
        model simple
            S1 -> S2; k1*S1
            S1 = 10; S2 = 0;
            k1 = 0.1;
        end
        """
        # create a RoadRunner object
        rr = tellurium.loada(model_str)
        # call the function to get the ODEs
        ode_str = utility.get_rr_model_ODEs(rr)
        # check that the ODEs are correct
        expected_ode_str = 'v_J0 = k1*S1\n\ndS1/dt = -v_J0\ndS2/dt = v_J0'
        self.assertEqual(ode_str.strip(), expected_ode_str.strip())


    def test_parse_config_file(self):
        '''Unit test to check if json file is correctly parsed'''
        # create a temporary json file for testing
        test_data = {'foo': 'bar', 'baz': 42}
        with open('test_config.json', 'w') as f:
            json.dump(test_data, f)

        # parse the test file
        result = utility.parse_config_file('test_config.json')

        # check that the returned dictionary matches the test data
        self.assertEqual(result, test_data)


class TestSSME(unittest.TestCase):
    def setUp(self):
        '''set up a roadrunner model and simulation keywords'''

        # create a RoadRunner model for testing
        self.model = roadrunner.RoadRunner('/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/scratch/antiporter_12D_model_3c_no_events.xml')

        # set up default arguments for simulate_y_pred_rr
        self.kwargs = {
            'n_points_stage': 500,
            'events': False,
            't_stage': 1,
            'buffer_labels': ['H_out', 'S_out'],
            'buffer_sequence': [(1e-7,1e-3),(5e-7,1e-3),(1e-7,1e-3)],
            'a_tol': 1e-22,
            'r_tol': 1e-12,
            'k_labels': ['k1_f','k1_r','k2_f','k2_r','k3_f' ,'k3_r' ,'k4_f' ,'k4_r' ,'k5_f' ,'k5_r' ,'k6_f'],
            'state_labels': ['OF', 'OF_Hb', 'IF_Hb', 'IF_Hb_Sb', 'IF_Sb', 'OF_Sb', 'H_in', 'S_in' ],
            'state_init': [ 0.0011694210430300167, 0, 0, 0, 0, 0, 1e-7, 1e-3],
            'output_label': ['time','current'],
        }
        

    def test_output_shape(self):
        '''Unit test to check if the simulation results have the correct shape'''
        # call simulate_y_pred_rr with default arguments
        y_pred = ssme.simulate_y_pred_rr(self.model, [1.0]*11, **self.kwargs)
        
        # check that the output has the expected shape
        self.assertEqual(y_pred.shape, (2, 1500))
        

    def test_output_type(self):
        '''Unit test to check if the simulation output has the correct type'''
        # call simulate_y_pred_rr with default arguments
        y_pred = ssme.simulate_y_pred_rr(self.model, [1.0]*11, **self.kwargs)
        
        # check that the output has the expected type
        self.assertIsInstance(y_pred, numpy.ndarray)


class TestSampler(unittest.TestCase):    
    def test_calc_normal_log_likelihood_exact(self):
        '''Unit test to check if the calculated log-likelihood matches expected results'''
        n = 1000
        sigma = 1.5
        y_pred = numpy.random.normal(0, 1, n)
        y_obs = y_pred + sigma*numpy.random.normal(0, 1, n)

        # Calculate analytical result
        analytical_result = -n/2*numpy.log(2*numpy.pi*sigma**2) - 1/(2*sigma**2)*numpy.sum((y_obs - y_pred)**2)

        # Calculate result using function
        result = sampler.calc_normal_log_likelihood(y_obs, y_pred, sigma)

        # Compare the two results
        assert numpy.isclose(result, analytical_result)


    def test_calc_normal_log_likelihood(self):
        '''Unit test to check if the calculated log-likelihood with true parameters is greater than with incorrect parameters'''
        # Create a random dataset with known parameters
        numpy.random.seed(42)
        true_mean = 2.0
        true_sigma = 0.5
        n_obs = 100
        y_obs = numpy.random.normal(loc=true_mean, scale=true_sigma, size=n_obs)
        
        # Calculate the log likelihood using the true parameters
        y_pred = numpy.ones_like(y_obs) * true_mean
        ll_true = sampler.calc_normal_log_likelihood(y_obs, y_pred, true_sigma)

        # Calculate the log likelihood using incorrect parameters
        y_pred_incorrect = numpy.ones_like(y_obs) * (true_mean + 1.0)
        ll_incorrect = sampler.calc_normal_log_likelihood(y_obs, y_pred_incorrect, true_sigma)

        # Verify that the log likelihood using true parameters is higher than using incorrect parameters
        assert ll_true > ll_incorrect


    def test_get_random_values_LHS(self):
        # Define parameter bounds
        param_bounds = [(0, 1), (1, 2), (2, 3)]
        # Generate 10 random parameter sets using LHS sampling
        samples = sampler.get_random_param_values(10, param_bounds, method='LHS')
        # Check that the shape of the output is correct
        self.assertEqual(samples.shape, (10, 3))
        # Check that all values are within the specified bounds
        self.assertTrue(numpy.all(samples >= numpy.array([0, 1, 2])))
        self.assertTrue(numpy.all(samples <= numpy.array([1, 2, 3])))


if __name__ == '__main__':
    unittest.main()