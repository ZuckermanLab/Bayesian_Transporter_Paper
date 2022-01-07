##### 6 parameter transporter model sampling - August George - 2021 

import numpy as np
from emcee import PTSampler
import tellurium as te
import pandas as pd
import time
import pathlib
import multiprocessing as mp
#import multiprocess as mp
#import ray.util.multiprocessing as mp
import matplotlib.pyplot as plt
import corner
import os
os.environ["OMP_NUM_THREADS"] = "1"










def main():



    def initialize_model(p):
        ''' get initial model (from random walker parameter set)'''
        k_conf = 10 ** p[0]
        k_H_on = 10 ** p[1]
        k_S_on = 10 ** p[2]
        k_H_off = 10 ** p[3]
        k_S_off = 10 ** p[4]
        _ = 10**p[5]
        # note: p[-1] is sigma (not used in transporter model calculation)


        antimony_string = f"""
                    // Created by libAntimony v2.12.0
                    model transporter_full()
        
                    // Compartments and Species:
                    compartment compartment_;
                    species $H_out in compartment_, OF in compartment_, OF_Hb in compartment_;
                    species IF_Hb in compartment_, S_in in compartment_, IF_Hb_Sb in compartment_;
                    species H_in in compartment_, IF_Sb in compartment_, OF_Sb in compartment_;
                    species $S_out in compartment_, IF in compartment_, OF_Hb_Sb in compartment_;
        
                    // Reactions:
                    rxn1: IF -> OF; compartment_*(rxn1_k1*IF - rxn1_k2*OF);
                    rxn2: OF + $H_out -> OF_Hb; compartment_*(rxn2_k1*OF*H_out - rxn2_k2*OF_Hb);
                    rxn3: OF_Sb -> OF + $S_out; compartment_*(rxn3_k1*OF_Sb - rxn3_k2*OF*S_out);
                    rxn4: OF_Hb -> IF_Hb; compartment_*(rxn4_k1*OF_Hb - rxn4_k2*IF_Hb);
                    rxn5: OF_Hb_Sb -> OF_Hb + $S_out; compartment_*(rxn5_k1*OF_Hb_Sb - rxn5_k2*OF_Hb*S_out);
                    rxn6: IF_Sb -> OF_Sb; compartment_*(rxn6_k1*IF_Sb - rxn6_k2*OF_Sb);
                    rxn7: OF_Sb + $H_out -> OF_Hb_Sb; compartment_*(rxn7_k1*OF_Sb*H_out - rxn7_k2*OF_Hb_Sb);
                    rxn8: OF_Hb_Sb -> IF_Hb_Sb; compartment_*(rxn8_k1*OF_Hb_Sb - rxn8_k2*IF_Hb_Sb);
                    rxn9: IF_Hb -> IF + H_in; compartment_*(rxn9_k1*IF_Hb - rxn9_k2*IF*H_in);
                    rxn10: IF + S_in -> OF_Sb; compartment_*(rxn10_k1*IF*S_in - rxn10_k2*OF_Sb);
                    rxn11: IF_Hb + S_in -> IF_Hb_Sb; compartment_*(rxn11_k1*IF_Hb*S_in - rxn11_k2*IF_Hb_Sb);
                    rxn12: IF_Hb_Sb -> IF_Sb + H_in; compartment_*(rxn12_k1*IF_Hb_Sb - rxn12_k2*IF_Sb*H_in);
        
                    // Events:
                    E1: at (time >= 5.0): H_out = 5e-8;
                    E2: at (time >= 10.0): H_out = 1e-7;
        
                    // Species initializations:
                    H_out = 1e-07;
                    H_out has substance_per_volume;
                    OF = 2.833e-8;
                    OF has substance_per_volume;
                    OF_Sb = 2.833e-8;
                    OF_Hb has substance_per_volume;
                    IF_Sb = 2.833e-8;
                    IF_Hb has substance_per_volume;
                    S_in = 1e-3;
                    S_in has substance_per_volume;
                    IF_Hb_Sb = 2.833e-8;
                    IF_Hb_Sb has substance_per_volume;
                    H_in = 1e-7;
                    H_in has substance_per_volume;
                    IF_Sb = 2.125e-08;
                    IF_Sb has substance_per_volume;
                    OF_Sb = 2.125e-08;
                    OF_Sb has substance_per_volume;
                    S_out = 0.001;
                    S_out has substance_per_volume;
                    IF = 0;
                    IF has substance_per_volume;
                    OF_Hb_Sb = 0;
                    OF_Hb_Sb has substance_per_volume;
        
        
                    // Compartment initializations:
                    compartment_ = 0.0001;
                    compartment_ has volume;
        
                    // Variable initializations:
                    k_conf = {k_conf};
                    k_S_on = {k_S_on};
                    k_S_off = {k_S_off};
                    k_H_on = {k_H_on};
                    k_H_off = {k_H_off};
        
                    // Variable initializations:
                    rxn1_k1 = 0;
                    rxn1_k2 = 0;
                    rxn2_k1 = k_H_on;
                    rxn2_k2 = k_H_off;
                    rxn3_k1 = k_S_off;
                    rxn3_k2 = k_S_on;
                    rxn4_k1 = k_conf;
                    rxn4_k2 = k_conf;
                    rxn5_k1 = 0;
                    rxn5_k2 = 0;
                    rxn6_k1 = k_conf;
                    rxn6_k2 = k_conf;
                    rxn7_k1 = 0;
                    rxn7_k2 = 0;
                    rxn8_k1 = 0;
                    rxn8_k2 = 0;
                    rxn9_k1 = 0;
                    rxn9_k2 = 0;
                    rxn10_k1 = 0;
                    rxn10_k2 = 0;
                    rxn11_k1 = k_S_on;
                    rxn11_k2 = k_S_off;
                    rxn12_k1 = k_H_off;
                    rxn12_k2 = k_H_on;
        
        
                    // Other declarations:
                    const compartment_, rxn1_k1, rxn1_k2, rxn2_k1, rxn2_k2, rxn3_k1, rxn3_k2;
                    const rxn4_k1, rxn4_k2, rxn5_k1, rxn5_k2, rxn6_k1, rxn6_k2, rxn7_k1, rxn7_k2;
                    const rxn8_k1, rxn8_k2, rxn9_k1, rxn9_k2, rxn10_k1, rxn10_k2, rxn11_k1;
                    const rxn11_k2, rxn12_k1, rxn12_k2, k_conf, k_S_on, k_H_on, k_S_off, k_H_off;
                    # const k_off;
        
                    // Unit definitions:
                    unit substance_per_volume = mole / litre;
                    unit volume = litre;
                    unit length = metre;
                    unit area = metre^2;
                    unit time_unit = second;
                    unit substance = mole;
                    unit extent = mole;
        
                    // Display Names:
                    time_unit is "time";
                    end
                    """ 

        z = te.loada(antimony_string)
        return z


    def get_y_pred(p,z):
        '''generates flux trace based on a set of parameters, p, and a Tellurium transporter model, z'''

        _ = 10**p[5]
        # reset z to initial
        z.resetToOrigin()

        # update theta
        k_conf_tmp = 10**p[0]
        k_H_on_tmp = 10**p[1]
        k_S_on_tmp = 10**p[2]
        k_H_off_tmp = 10**p[3]
        k_S_off_tmp = 10**p[4]
        z.k_conf = k_conf_tmp
        z.k_H_on = k_H_on_tmp
        z.k_S_on = k_S_on_tmp
        z.k_H_off= k_H_off_tmp
        z.k_S_off= k_S_off_tmp
        z.rxn2_k1 = k_H_on_tmp
        z.rxn2_k2 = k_H_off_tmp
        z.rxn3_k1 = k_S_off_tmp
        z.rxn3_k2 = k_S_on_tmp
        z.rxn4_k1 = k_conf_tmp
        z.rxn4_k2 = k_conf_tmp
        z.rxn6_k1 = k_conf_tmp
        z.rxn6_k2 = k_conf_tmp
        z.rxn11_k1 = k_S_on_tmp
        z.rxn11_k2 = k_S_off_tmp
        z.rxn12_k1 = k_H_off_tmp
        z.rxn12_k2 = k_H_on_tmp

        

        # set tolerances for simulations
        z.integrator.absolute_tolerance = 1e-19
        z.integrator.relative_tolerance = 1e-17

        n_stage = 3  # number of stages: equilibration, activation, reversal
        t_stage = 5  # time length for each stage (in sec) to allow for equilibration
        n_iter_stage = 5e3  # how many how many ODE solver iterations per stage
        t_res = 0.04  # time resolution (sec)
        n_samples_stage = int(t_stage / t_res)  # how many data points per stage

        t_0 = 0
        t_f = int(np.floor(n_stage * t_stage))
        n_iter = int(np.floor(n_iter_stage * n_stage))
        idx_s2 = int(np.floor(n_iter_stage))
        step_size = int(np.floor(n_iter_stage / n_samples_stage))

        try:        
            D = pd.DataFrame(z.simulate(t_0, t_f, n_iter, selections=['time', 'rxn9', 'rxn12']),
                            columns=['time', 'rxn9', 'rxn12'])
            y_calc = pd.DataFrame(D['rxn9'] + D['rxn12'], columns=['y_calc'])  # rxn 9 + rxn 12
            y_pred = y_calc.iloc[idx_s2+4:2*idx_s2:step_size]
            #y_pred = y_calc.iloc[idx_s2::step_size]
        except:
            print('error in y_pred calculations')
            y_error_list = [0 * i for i in range(125)]  # datapoints?
            y_pred = pd.DataFrame(y_error_list, columns='y_calc')
            with open(f'emcee_transporter_error.txt', 'a') as f:
                f.write(f'\nERROR IN Y_PRED CALCULATIONS!\nSetting y_pred={y_error_list}\n')
        return y_pred['y_calc'].values



    def log_likelihood(theta, y_obs, model):
        '''log of Guassian likelihood distribution'''
        #print(theta)
        curr_sigma = 10**theta[5]
        y_pred = get_y_pred(theta, model)

        # calculate normal log likelihood
        logl = -len(y_obs) * np.log(np.sqrt(2.0 * np.pi) * curr_sigma)
        logl += -np.sum((y_obs - y_pred) ** 2.0) / (2.0 * curr_sigma ** 2.0) 
        return logl


    def log_prior(theta):
        '''log of uniform prior distribution'''

        k_conf = theta[0]
        k_H_on = theta[1]
        k_S_on = theta[2]
        k_H_off = theta[3]
        k_S_off = theta[4]
        sigma = theta[5]

        # if prior is between boundary --> log(prior) = 0 (uninformitive prior)
        if np.log10(5e-14)<sigma<np.log10(5e-13) and -1<k_conf<5 and 7<k_H_on<13 and 4<k_S_on<10 \
            and 0<k_H_off<6 and 0<k_S_off<6:
            return 0  
        else:
            return -np.inf


    def log_probability(theta, y_obs, model):
        '''log of estimated posterior probability'''
        logp = log_prior(theta)
        if not np.isfinite(logp):
            return -np.inf  # ~zero probability
        return logp + log_likelihood(theta, y_obs, model)  # log posterior ~ log likelihood + log prior


    # global scope - fix later
    datafile = 'emcee_transporter_data.csv'
    y_obs = np.loadtxt(f'{datafile}', delimiter=',', skiprows=1, usecols=1).tolist()  # load data from file
    ##### initial model #####
    sigma_init = np.random.uniform(np.log10(5e-14),np.log10(5e-13))
    k_conf_init = np.random.uniform(-1,5)
    k_H_on_init = np.random.uniform(7,13)
    k_S_on_init = np.random.uniform(4,10)
    k_H_off_init = np.random.uniform(0,6)
    k_S_off_init = np.random.uniform(0,6)
    p_init = [k_conf_init,k_H_on_init,k_S_on_init,k_H_off_init,k_S_off_init,sigma_init]
    model = initialize_model(p_init)
    #

    ##### sampling settings #####
    n_replicas = 1  # repeats using different starting points 
    n_walkers = 12  # at least 3x the number of parameters
    n_steps = int(1e3)  # at least 50x the autocorrelation time
    n_burn = int(0.1*n_steps)
    n_temps = 10
    seed = 1234
    parallel=False
    batch_sampling=False
    np.random.seed(seed)



    ##### synthetic model settings #####
    datafile = 'emcee_transporter_data.csv'
    labels = ["k_conf", "k_H_on", "k_S_on", "k_H_off", "k_S_off", "sigma"]  
    theta_true = [1e2, 1e10, 1e7, 1e3, 1e3, 1e-13]  # theta and labels must be in same order
    print('debug: parallelization bug')
    theta_true_log = np.log10(theta_true)
    theta_ref = theta_true_log
    n_dim = len(theta_true)
    #y_obs = np.loadtxt(f'{datafile}', delimiter=',', skiprows=1, usecols=1).tolist()  # load data from file
    

    ##### initial model #####
    # sigma_init = np.random.uniform(np.log10(5e-14),np.log10(5e-13))
    # k_conf_init = np.random.uniform(-1,5)
    # k_H_on_init = np.random.uniform(7,13)
    # k_S_on_init = np.random.uniform(4,10)
    # k_H_off_init = np.random.uniform(0,6)
    # k_S_off_init = np.random.uniform(0,6)
    # p_init = [k_conf_init,k_H_on_init,k_S_on_init,k_H_off_init,k_S_off_init,sigma_init]

    # model = initialize_model(p_init)
    y_true = get_y_pred(theta_ref,model)



    ##### output settings #####
    time_str = time.strftime("%Y%m%d_%H%M%S") 
    new_dir = pathlib.Path(pathlib.Path.cwd(), f'{time_str}_emcee_transporter')
    new_dir.mkdir(parents=True, exist_ok=True)
    with open(new_dir/f'{time_str}_emcee_transporter_log.txt', 'a') as f:
        f.write(f'{time_str}_emcee_transporter_log.txt\n\n')
        f.write(f'timestamp:{time_str}\n')
        f.write(f'n replicas:{n_replicas}\nn walkers:{n_walkers}\nn steps/walker:{n_steps}\n' )
        f.write(f'datafile:{datafile}\nparameters:{labels}\nparameter reference values:{theta_true}\n')


    ##### parallel tempering sampling
    sampler=PTSampler(n_temps, n_walkers, n_dim, log_likelihood, log_prior, loglargs=[y_obs, model] )
    # random starts from uniform priors
    p0_list = []
    for j in range(n_temps):
        pos_list = []
        for i in range(n_walkers):
            sigma_i = np.random.uniform(np.log10(5e-14),np.log10(5e-13))
            k_conf_i = np.random.uniform(-1,5)
            k_H_on_i = np.random.uniform(7,13)
            k_S_on_i = np.random.uniform(4,10)
            k_H_off_i = np.random.uniform(0,6)
            k_S_off_i = np.random.uniform(0,6)
            pos_list.append([k_conf_i,k_H_on_i,k_S_on_i,k_H_off_i,k_S_off_i,sigma_i])
        p0_list.append(pos_list)
    p0 = np.asarray(p0_list)
    assert(np.shape(p0) == (n_temps,n_walkers,n_dim))

    i=0
    for p, lnprob, lnlike in sampler.sample(p0, iterations=n_burn):
        print(f'{i+1}/{n_burn}')
        i+=1
    sampler.reset()
    i=0
    for p, lnprob, lnlike in sampler.sample(p, lnprob0=lnprob,
                                            lnlike0=lnlike,
                                            iterations=n_steps):
        print(f'{i+1}/{n_steps}')
        i+=1
        pass

    assert sampler.chain.shape == (n_temps, n_walkers, n_steps, n_dim)

    # Chain has shape (ntemps, nwalkers, nsteps, ndim)
    # Zero temperature mean:
    mu0 = np.mean(np.mean(sampler.chain[0,...], axis=0), axis=0)

    # Longest autocorrelation length (over any temperature)
    #print(sampler.acor)
    #max_acl = np.max(sampler.acor)
    #labels = ['x1', 'x2']
    flat_samples = sampler.flatchain[0,:,:]
    fig = corner.corner(
                    flat_samples, bins=100, labels=labels)
    plt.suptitle('low T')
    plt.savefig(new_dir/'pair_low_t.png')

    flat_samples2 = sampler.flatchain[-1,:,:]
    fig = corner.corner(
                    flat_samples2, bins=100, labels=labels)
    plt.suptitle('high T')
    plt.savefig(new_dir/'pair_high_t.png')



    flat_samples3 = sampler.flatchain[2,:,:]
    fig = corner.corner(
                    flat_samples3, bins=100, labels=labels)
    plt.suptitle('mid T')
    plt.savefig(new_dir/'pair_mid_t.png')
   


        
if __name__ == '__main__':

    main()