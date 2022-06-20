import AIS_affine as aisa
import unittest
import tellurium as te
import numpy as np
import logging
import os
import sys
import ray
import time
import emcee
import ray.util.multiprocessing as mp
import copy
#import multiprocessing as mp


class TestAISAffine12D(unittest.TestCase):
    def setUp(self):
        self.seed = 42
        self.rng = np.random.default_rng(self.seed)
        #np.random.seed(seed)
        self.n_cpus = 6

        with open("antiporter_12D_model.txt", "r") as f:
            antimony_string_SS = f.read()
        self.m = te.loada(antimony_string_SS)
        self.m.integrator.absolute_tolerance = 1e-18
        self.m.integrator.relative_tolerance = 1e-12
        self.m.H_out = 5e-8
        D1 = self.m.simulate(0, 5, 125, selections=['time', 'rxn4'])
        y_true = D1['rxn4'][1:]  # remove first point

        self.noise_stdev_true = 1e-13
        self.y_obs = np.genfromtxt("data_grid_test3_1exp_v2.csv")
        self.p_info = [   
            ["log_k1_f",6,12,10],
            ["log_k1_r",-1,5,3],
            ["log_k2_f",-2,4,2],
            ["log_k2_r",-2,4,2],
            ["log_k3_f",3,9,7],
            ["log_k3_r",-1,5,3],
            ["log_k4_f",-1,5,3],
            ["log_k4_r",6,12,10],
            ["log_k5_f",-2,4,2],
            ["log_k5_r",-2,4,2],
            ["log_k6_f",-1,5,3],
            ["log_sigma",np.log10(5e-14), np.log10(5e-13), -13],
        ]
        self.param_ref = [p_i[3] for p_i in self.p_info]
        self.labels = [p[0] for p in self.p_info]
        self.b_list = [(p[1], p[2]) for p in self.p_info]

        for b in self.b_list:
                # b[0] = lower bound for prior, b[1] = upper bound for prior
                # lower bound should be less than upper bound
                assert(b[0]<b[1])

        self.dim = 12
        self.max_iter = 1e5
        self.save_at = int(10)  # save data every x steps
        
        self.fractional_weight_target = 0.65
        self.n_init_samples = int(1e4)
        self.N_steps = int(1e3)
        self.K_walkers = int(1e3)
        self.NK_total_samples = self.N_steps*self.K_walkers
        self.burn_in = int(0.1*self.NK_total_samples)
        self.init_beta_jump_target = 0.0001
        #self.init_batch_size = int(self.n_init_samples/self.n_cpus)
        self.init_batch_size = int(self.n_init_samples*0.1)
        #init_ESS = K_walkers
        self.init_fractional_weight_target = self.K_walkers/self.n_init_samples
        self.M_sub_samples = self.NK_total_samples - self.burn_in #10*K_walkers
        self.beta=0.0

        # number of subsamples is should be less than total number of samples
        assert(self.M_sub_samples < self.NK_total_samples)

        # number of subsamples is greater than or equal to 2x the number of walkers (heuristic)
        assert(self.M_sub_samples >= 2*self.K_walkers)
        

    def testInit(self):
        """creating a dense initial set of samples in the sampling space"""
        rng1 = np.random.default_rng(1)
        rng2 = np.random.default_rng(2)
        samples = aisa.get_p0(self.b_list, self.n_init_samples, rng1)
        samples2 = aisa.get_p0(self.b_list, self.n_init_samples, rng2)

        # samples should have correct shape
        assert(np.shape(samples)==(self.n_init_samples, self.dim))
        assert(np.shape(samples2)==(self.n_init_samples, self.dim))

        # sample set 1 and 2 should not be equal w/ different rng seeds
        assert(not np.array_equal(samples, samples2))
        assert(not np.array_equiv(samples, samples2))


    def testInitBatch(self):
        """creating a dense initial set of samples in the sampling space - using a batching approach"""
        rng1 = np.random.default_rng(1)
        rng2 = np.random.default_rng(1)
        samples = aisa.get_p0(self.b_list, self.n_init_samples, rng1)
        batch_samples = aisa.get_p0_batch(self.b_list, self.n_init_samples, self.init_batch_size, rng2)

        # samples should have correct shape  
        assert(np.shape(samples)==(self.n_init_samples, self.dim))
        assert(np.shape(batch_samples)==(self.n_init_samples, self.dim))
  
        # samples using batching should be the same as w/o batching
        # assert(np.array_equal(samples, batch_samples))  # not working, probably something with rng state

    ## parallelization is slow - refactor batching strategy
    # def testBatchInitRay(self):
    #     ray.init(num_cpus=self.n_cpus)

    #     rng1 = np.random.default_rng(1)
    #     rng2 = np.random.default_rng(1)
    #     t0 = time.time()
    #     samples = aisa.get_p0(self.b_list, self.n_init_samples, rng1)
    #     t1 = time.time()
    #     batch_samples = aisa.get_p0_batch(self.b_list, self.n_init_samples, self.init_batch_size, rng2)
    #     t2 = time.time()
    #     batch_samples_ray = aisa.get_p0_batch_ray(self.b_list, self.n_init_samples, self.init_batch_size, rng2)
    #     t3 = time.time()

    #     print(t1-t0)
    #     print(t2-t1)
    #     print(t3-t2)
  
    #     # samples should have correct shape
    #     assert(np.shape(samples)==(self.n_init_samples, self.dim))
    #     assert(np.shape(batch_samples)==(self.n_init_samples, self.dim))
    #     assert(np.shape(batch_samples_ray)==(self.n_init_samples, self.dim))

    #     # samples using batching should be the same as w/o batching
    #     # assert(np.array_equal(samples, batch_samples))  # not working, probably something with rng state
    #     ray.shutdown()


    def testInitSampling(self):
        """calculating loglikelihood, next beta, relative probabilities and sampling during the initial step"""
        n = int(1e4)
        beta = 0.0
        init_fractional_weight_target = 0.65
        rng1 = np.random.default_rng(1)
        samples = aisa.get_p0(self.b_list, n, rng1)
        log_like = np.nan_to_num(np.array([aisa.calc_log_prob(theta_i,self.y_obs,[self.m,1]) for theta_i in samples])) 
        ref_likelihood = aisa.calc_log_prob(self.param_ref, self.y_obs, [self.m,1])
        beta_new = aisa.calculate_next_beta(log_like, beta, init_fractional_weight_target)
        
        print(beta)
        print(f"beta new: {beta_new}")
        print(init_fractional_weight_target)
        

        p_rel = aisa.calculate_p_rel(log_like, beta, beta_new)
        resamples_index = self.rng.choice([ i for i in range(len(samples))],size=self.K_walkers,p=p_rel)
        resamples = samples[resamples_index]
        p0 = np.array([s for s in resamples])
        sampler = emcee.EnsembleSampler(self.K_walkers, self.dim, aisa.calc_log_prob, args=[self.y_obs, [self.m,beta_new]])
        state = sampler.run_mcmc(p0, self.N_steps)

        samples = sampler.flatchain[self.burn_in:]
        print(np.shape(samples))
        print(np.size(samples))

        assert(np.isnan(log_like).any()==False and np.isinf(log_like).any()==False) # no NaN or Inf
        assert(np.size(log_like)==n)
        assert(ref_likelihood>1000)  # ref likelihood should be large and positive (empirical number)
        assert(np.max(log_like)>=0.1*ref_likelihood)  # initial sampled log likelihood should include value near reference
        assert(beta_new >=0)  # next beta should be > 0
        assert(beta_new <=1)  # next beta should not be 1 with large overlap
        assert(np.sum(p_rel)==1)  # sum of probabilities = 1
        assert(np.size(p_rel)==n)
        assert(np.shape(samples)[0]==self.M_sub_samples)


    def testInitSamplingBatch(self):
        """calculating loglikelihood, next beta, relative probabilities and sampling during the initial step - using a batch approach"""
                    # batch_list_refs = []
            # for s_i in range(0,n_init_samples, init_batch_size):
            #     y_obs_list = [y_obs for i in range(n_init_samples)]
            #     extra_args_list = [[m,1] for i in range(n_init_samples)]
            #     batch = samples[s_i : s_i + init_batch_size] 
            #     batch_args = list(zip(batch, y_obs_list,extra_args_list))
            #     batch_list_refs.append(log_prob_ray_batch.remote(batch_args))
            # log_like = np.nan_to_num(np.array(ray.get(batch_list_refs)))

            #ray.shutdown()
        raise NotImplementedError

    def testInitSamplingBatchParallel(self):
        """calculating loglikelihood, next beta, relative probabilities and sampling during the initial step - using a batch approach and parallelization"""
        raise NotImplementedError

    def testSampling(self):
        """sampling during subsequent steps"""
        """sampling during subsequent steps - with parallelization"""
        ray.init(num_cpus=self.n_cpus, object_store_memory=2*10**9)
        p0 = aisa.get_p0(self.b_list, self.K_walkers, self.rng)
        assert(np.shape(p0)==(self.K_walkers, self.dim))
        
        # serial version
        t1 = time.time()
        sampler = emcee.EnsembleSampler(self.K_walkers, self.dim, aisa.calc_log_prob, args=[self.y_obs, [self.m,1]])
        state = sampler.run_mcmc(p0, self.N_steps)
        samples = sampler.flatchain[self.burn_in:]
        assert(np.shape(samples)[0]==self.M_sub_samples)
        
        


    # def testSamplingBatch(self):
    #     raise NotImplementedError

    def testSamplingParallel(self):
        """sampling during subsequent steps - with parallelization"""
        ray.init(num_cpus=self.n_cpus, object_store_memory=2*10**9)
        p0 = aisa.get_p0(self.b_list, self.K_walkers, self.rng)
        assert(np.shape(p0)==(self.K_walkers, self.dim))
        
        # serial version
        t1 = time.time()
        sampler = emcee.EnsembleSampler(self.K_walkers, self.dim, aisa.calc_log_prob, args=[self.y_obs, [self.m,1]])
        state = sampler.run_mcmc(p0, self.N_steps)
        samples = sampler.flatchain[self.burn_in:]
        assert(np.shape(samples)[0]==self.M_sub_samples)
        t2 = time.time()

        # parallel version
        with mp.Pool(processes=self.n_cpus) as pool:
            sampler2 = emcee.EnsembleSampler(self.K_walkers, self.dim, aisa.calc_log_prob, args=[self.y_obs, [self.m,1]], pool=pool)
            state2 = sampler2.run_mcmc(p0, self.N_steps)
        samples2 = sampler2.flatchain[self.burn_in:]
        assert(np.shape(samples2)[0]==np.shape(samples)[0])
        t3 = time.time()
        aisa.plot_samples(samples,self.p_info, 1, [self.K_walkers, self.K_walkers, self.N_steps, self.fractional_weight_target, self.seed, "9"])
        aisa.plot_samples(samples2,self.p_info, 1, [self.K_walkers, self.K_walkers, self.N_steps, self.fractional_weight_target, self.seed, "10"])
        print(f'serial {t2-t1}')
        print(f'parallel {t3-t2}')
        import pandas as pd
        df = pd.DataFrame(samples, columns=[i[0] for i in self.p_info])
        df.to_csv(f'TEST_ais_affine_{self.n_init_samples}i_{self.K_walkers}w_{self.N_steps}s_{self.fractional_weight_target}t_{self.seed}r_data.csv')
        
        



# class TestAISAffine1D(unittest.TestCase):
#     def setUp(self):
#         """setup a 1D bimodal Gaussian with known analytical solution for correctness testing"""
#         # create 1D model w/ known value
#         raise NotImplementedError

#     def testInit(self):
#         """creating a dense initial set of samples in the sampling space"""
#         raise NotImplementedError

#     def testInitBatch(self):
#         """creating a dense initial set of samples in the sampling space - using a batching approach"""
#         raise NotImplementedError

#     def testInitBatchParallel(self):
#         """creating a dense initial set of samples in the sampling space - using a batching approach and parallelization"""
#         raise NotImplementedError

#     def testInitSampling(self):
#         """calculating loglikelihood, next beta, relative probabilities and sampling during the initial step"""
#         raise NotImplementedError

#     def testInitSamplingBatch(self):
#         """calculating loglikelihood, next beta, relative probabilities and sampling during the initial step - using a batching approach"""
#         raise NotImplementedError

#     def testInitSamplingBatchParallel(self):
#         """calculating loglikelihood, next beta, relative probabilities and sampling during the initial step - using a batching approach and parallelization"""
#         raise NotImplementedError

#     def testSampling(self):
#         """sampling during subsequent steps"""
#         raise NotImplementedError

#     def testSamplingParallel(self):
#         """sampling during subsequent steps - with parallelization"""
#         raise NotImplementedError


if __name__ == "__main__":
    unittest.main()