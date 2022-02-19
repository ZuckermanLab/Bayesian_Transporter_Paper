from re import X
import tellurium as te
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import corner
import pandas as pd
import time
import pathlib 
import os
from datetime import datetime


# use less synthetic data (removes tails)
less_data = True
data_amount = 0.5 # 0.5 = keep 50% of data per experiment stage

# use 3 experiments
three_exp = False




# check environment (fix this later)
use_pt_sampler = True
conda_env = os.environ['CONDA_PREFIX']
print(f'pt sampler={use_pt_sampler}')
print(conda_env)


if use_pt_sampler == True:
    if conda_env != '/opt/miniconda3/envs/pyPT':
        print('error: not using correct conda environment for parallel tempering')
        exit()
if use_pt_sampler == False:
    if conda_env != '/opt/miniconda3/envs/BayesGrid':
        print('error: not using correct environment traditional emcee')
        exit()

import emcee as mc



def init_model(p):

    ### note: p[-1] is sigma -not rate constant!

    theta = [10**i for i in p]
    antimony_string = f"""
            // Created by libAntimony v2.12.0
            model transporter_full()

            // Compartments and Species:
            compartment vol;
            species $H_out in vol, OF in vol, OF_Hb in vol;
            species IF_Hb in vol, S_in in vol, IF_Hb_Sb in vol;
            species H_in in vol, IF_Sb in vol, OF_Sb in vol;
            species $S_out in vol, IF in vol, OF_Hb_Sb in vol;

            // Reactions:
            rxn1: IF -> OF; vol*(rxn1_k1*IF - rxn1_k2*OF);
            rxn2: OF + $H_out -> OF_Hb; vol*(rxn2_k1*OF*H_out - rxn2_k2*OF_Hb);
            rxn3: OF_Sb -> OF + $S_out; vol*(rxn3_k1*OF_Sb - rxn3_k2*OF*S_out);
            rxn4: OF_Hb -> IF_Hb; vol*(rxn4_k1*OF_Hb - rxn4_k2*IF_Hb);
            rxn5: OF_Hb_Sb -> OF_Hb + $S_out; vol*(rxn5_k1*OF_Hb_Sb - rxn5_k2*OF_Hb*S_out);
            rxn6: IF_Sb -> OF_Sb; vol*(rxn6_k1*IF_Sb - rxn6_k2*OF_Sb);
            rxn7: OF_Sb + $H_out -> OF_Hb_Sb; vol*(rxn7_k1*OF_Sb*H_out - rxn7_k2*OF_Hb_Sb);
            rxn8: OF_Hb_Sb -> IF_Hb_Sb; vol*(rxn8_k1*OF_Hb_Sb - rxn8_k2*IF_Hb_Sb);
            rxn9: IF_Hb -> IF + H_in; vol*(rxn9_k1*IF_Hb - rxn9_k2*IF*H_in);
            rxn10: IF + S_in -> IF_Sb; vol*(rxn10_k1*IF*S_in - rxn10_k2*IF_Sb);
            rxn11: IF_Hb + S_in -> IF_Hb_Sb; vol*(rxn11_k1*IF_Hb*S_in - rxn11_k2*IF_Hb_Sb);
            rxn12: IF_Hb_Sb -> IF_Sb + H_in; vol*(rxn12_k1*IF_Hb_Sb - rxn12_k2*IF_Sb*H_in);

            // Events:
            E1: at (time >= 5.0): H_out = H_out_activation, S_out = S_out_activation;
            E2: at (time >= 10.0): H_out = 1e-7, S_out = 0.001;

            // Species initializations:
            H_out = 1e-07;
            H_out has substance_per_volume;

            H_in = 1e-7;
            H_in has substance_per_volume;

            S_out = 0.001;
            S_out has substance_per_volume;

            S_in = 1e-3;
            S_in has substance_per_volume;

            OF = 2.833e-8;
            OF has substance_per_volume;

            IF = 2.125e-08;;
            IF has substance_per_volume;

            OF_Hb = 2.833e-8;
            OF_Hb has substance_per_volume;

            IF_Hb = 2.833e-8;
            IF_Hb has substance_per_volume;

            OF_Sb = 2.125e-08;
            OF_Sb has substance_per_volume;

            IF_Sb = 2.125e-08;
            IF_Sb has substance_per_volume;

            OF_Hb_Sb = 2.125e-08;;
            OF_Hb_Sb has substance_per_volume;

            IF_Hb_Sb = 2.833e-8;
            IF_Hb_Sb has substance_per_volume;

            // Compartment initializations:
            vol = 0.0001;
            vol has volume;

            // Variable initializations:
            H_out_activation = 5e-8;
            S_out_activation = 0.001;

            // Rate constant initializations:
            rxn1_k1 = 0;
            rxn1_k2 = 0;
            rxn2_k1 = {theta[0]};
            rxn2_k2 = {theta[1]};
            rxn3_k1 = {theta[2]};
            rxn3_k2 = {theta[3]};
            rxn4_k1 = {theta[4]};
            rxn4_k2 = {theta[5]};
            rxn5_k1 = 0;
            rxn5_k2 = 0;
            rxn6_k1 = {theta[6]};
            rxn6_k2 = {theta[7]};
            rxn7_k1 = 0;
            rxn7_k2 = 0;
            rxn8_k1 = 0;
            rxn8_k2 = 0;
            rxn9_k1 = 0;
            rxn9_k2 = 0;
            rxn10_k1 = 0;
            rxn10_k2 = 0;
            rxn11_k1 = {theta[8]};
            rxn11_k2 = {theta[9]};
            rxn12_k1 = {theta[10]};
            rxn12_k2 = {(theta[0]*theta[2]*theta[4]*theta[6]*theta[8]*theta[10])/(theta[1]*theta[3]*theta[5]*theta[7]*theta[9])};

            rxn12: IF_Hb_Sb -> IF_Sb + H_in; vol*(rxn12_k1*IF_Hb_Sb - rxn12_k2*IF_Sb*H_in);

            // Other declarations:
            const vol, rxn1_k1, rxn1_k2, rxn2_k1, rxn2_k2, rxn3_k1, rxn3_k2;
            const rxn4_k1, rxn4_k2, rxn5_k1, rxn5_k2, rxn6_k1, rxn6_k2, rxn7_k1, rxn7_k2;
            const rxn8_k1, rxn8_k2, rxn9_k1, rxn9_k2, rxn10_k1, rxn10_k2, rxn11_k1;
            const rxn11_k2, rxn12_k1, rxn12_k2

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


def simulate_model(p, z):
    '''generates flux trace based on a set of parameters, p, and a Tellurium transporter model, z'''

    # percentage of data to keep 
    sf_1 = data_amount  # experiment 1 (.3)
    sf_2 = data_amount  # experiment 2 (.5)
    sf_3 = data_amount   # experiment 3

    # reset z to initial
    z.resetToOrigin()

    #update pH
    z.H_out_activation = 5e-8

    # update rate constants

    z.rxn2_k1 = 10**p[0]
    z.rxn2_k2 = 10**p[1]
    z.rxn3_k1 = 10**p[2]
    z.rxn3_k2 = 10**p[3]
    z.rxn4_k1 = 10**p[4]
    z.rxn4_k2 = 10**p[5]
    z.rxn6_k1 = 10**p[6]
    z.rxn6_k2 = 10**p[7]
    z.rxn11_k1 = 10**p[8]
    z.rxn11_k2 = 10**p[9]
    z.rxn12_k1 = 10**p[10]
   
    # cycle 1 constraint
    c1_fwd = (10**p[0])*(10**p[2])*(10**p[4])*(10**p[6])*(10**p[8])*(10**p[10])
    c1_rev_wo_rxn12_k2 =  (10**p[1])*(10**p[3])*(10**p[5])*(10**p[7])*(10**p[9])
    z.rxn12_k2 = c1_fwd/c1_rev_wo_rxn12_k2
    assert(np.isclose((c1_fwd/(c1_rev_wo_rxn12_k2*z.rxn12_k2)),1))

    
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
        D = z.simulate(t_0, t_f, n_iter, selections=['time', 'rxn9', 'rxn12'])
        y_calc = D['rxn9']+D['rxn12']
        #s1 = y_calc[:idx_s2+4:step_size]
       
     
        s2 = y_calc[idx_s2+4:2*idx_s2:step_size]
        s3 = y_calc[2*idx_s2+4::step_size]

        if less_data:
            
            s2 = s2[:int(len(s2)*sf_1)]
            s3 = s3[:int(len(s3)*sf_1)]

        y_pred_1 = np.hstack([s2,s3])
    except:
        print('simulation error')
        return np.zeros(500)


    ### experiment 2

    # reset z to initial
    z.resetToOrigin()

    #update pH
    z.H_out_activation = 5e-7

    # update rate constants
    z.rxn2_k1 = 10**p[0]
    z.rxn2_k2 = 10**p[1]
    z.rxn3_k1 = 10**p[2]
    z.rxn3_k2 = 10**p[3]
    z.rxn4_k1 = 10**p[4]
    z.rxn4_k2 = 10**p[5]
    z.rxn6_k1 = 10**p[6]
    z.rxn6_k2 = 10**p[7]
    z.rxn11_k1 = 10**p[8]
    z.rxn11_k2 = 10**p[9]
    z.rxn12_k1 = 10**p[10]
    #z.rxn12_k2 = 10**p[11]

    # cycle 1 constraint
    c1_fwd = (10**p[0])*(10**p[2])*(10**p[4])*(10**p[6])*(10**p[8])*(10**p[10])
    c1_rev_wo_rxn12_k2 =  (10**p[1])*(10**p[3])*(10**p[5])*(10**p[7])*(10**p[9])
    z.rxn12_k2 = c1_fwd/c1_rev_wo_rxn12_k2
    assert(np.isclose((c1_fwd/(c1_rev_wo_rxn12_k2*z.rxn12_k2)),1))

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
        D2 = z.simulate(t_0, t_f, n_iter, selections=['time', 'rxn9', 'rxn12'])
        y_calc2 = D2['rxn9']+D2['rxn12']

        #s1_2 = y_calc2[:idx_s2+4:step_size]
        s2_2 = y_calc2[idx_s2+4:2*idx_s2:step_size]
        s3_2 = y_calc2[2*idx_s2+4::step_size]
        if less_data:
            s2_2 = s2_2[:int(len(s2_2)*sf_2)]
            s3_2 = s3_2[:int(len(s3_2)*sf_2)]

        y_pred_2 = np.hstack([s2_2,s3_2])
        if three_exp == False:
            y_pred = np.hstack([y_pred_1, y_pred_2])
    except:
        print('error in simulations')
        return np.zeros(500)

    ### experiment 3
    if three_exp == True:
        # reset z to initial
        z.resetToOrigin()

        #update pH
        z.H_out_activation = 5e-7
        z.S_out_activation = 0.002

        # update rate constants
        z.rxn2_k1 = 10**p[0]
        z.rxn2_k2 = 10**p[1]
        z.rxn3_k1 = 10**p[2]
        z.rxn3_k2 = 10**p[3]
        z.rxn4_k1 = 10**p[4]
        z.rxn4_k2 = 10**p[5]
        z.rxn6_k1 = 10**p[6]
        z.rxn6_k2 = 10**p[7]
        z.rxn11_k1 = 10**p[8]
        z.rxn11_k2 = 10**p[9]
        z.rxn12_k1 = 10**p[10]
        #z.rxn12_k2 = 10**p[11]

        # cycle 1 constraint
        c1_fwd = (10**p[0])*(10**p[2])*(10**p[4])*(10**p[6])*(10**p[8])*(10**p[10])
        c1_rev_wo_rxn12_k2 =  (10**p[1])*(10**p[3])*(10**p[5])*(10**p[7])*(10**p[9])
        z.rxn12_k2 = c1_fwd/c1_rev_wo_rxn12_k2
        assert(np.isclose((c1_fwd/(c1_rev_wo_rxn12_k2*z.rxn12_k2)),1))

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
            D3 = z.simulate(t_0, t_f, n_iter, selections=['time', 'rxn9', 'rxn12'])
            y_calc3 = D3['rxn9']+D3['rxn12']
            #s1_2 = y_calc2[:idx_s2+4:step_size]
            s2_3 = y_calc3[idx_s2+4:2*idx_s2:step_size]
            s3_3 = y_calc3[2*idx_s2+4::step_size]
            if less_data:
                s2_3 = s2_3[:int(len(s2_3)*sf_3)]
                s3_3 = s3_3[:int(len(s3_3)*sf_3)]

            y_pred_3 = np.hstack([s2_3,s3_3])
            y_pred = np.hstack([y_pred_1, y_pred_2, y_pred_3])
        except:
            print('error in simulations')
            return np.zeros(500)

    return y_pred


def log_likelihood(theta, y_obs, model):
    '''log of Guassian likelihood distribution'''
    #print(theta)
    #curr_sigma = 10**theta[-1]
    curr_sigma = theta[-1]
    
    y_pred = simulate_model(theta, model)



    # calculate normal log likelihood
  
    logl = -len(y_obs) * np.log(np.sqrt(2.0 * np.pi) * curr_sigma)
   
    logl += -np.sum((y_obs - y_pred) ** 2.0) / (2.0 * curr_sigma ** 2.0) 
  
    return logl


def log_prior(theta):
    '''log of uniform prior distribution'''
    # if prior is between boundary --> log(prior) = 0 (uninformitive prior)

    valid_prior=check_prior(theta)
    if valid_prior == True:
        return 0
    else:
        return -np.inf
    

def log_probability(theta, y_obs, model):
    '''log of estimated posterior probability'''
    logp = log_prior(theta)
    if not np.isfinite(logp):
        return -np.inf  # ~zero probability
    return logp + log_likelihood(theta, y_obs, model)  # log posterior ~ log likelihood + log prior


def randomize_model_parameters(p, s=0):
    '''randomize initial parameter values to be within accepted range for each parameter type'''
   


    # k_conf_range = (-1-s,5)
    # k_H_on_range = (7-s,13)
    # k_H_off_range = (0-s,6)
    # k_S_on_range = (4-s,10)
    # k_S_off_range = (0-s,6)
    # # sigma_range = (np.log10(5e-14),np.log10(5e-13))
    # sigma_range = ((1e-13 - (1e-13*0.5)), (1e-13 + (1e-13*0.5)))


    # prior shift (testing)
    #s = [-1,1, -1, 1,-1]
    # s = [-0.5, 0.5, -0.5, 0.5, -0.5]
    s = [0, 0, 0, 0, 0,]
    k_conf_range = (-1-s[0],5-s[0])
    k_H_on_range = (7-s[1],13-s[1])
    k_H_off_range = (0-s[2],6-s[2])
    k_S_on_range = (4-s[3],10-s[3])
    k_S_off_range = (0-s[4],6-s[4])
    sigma_range = ((1e-13 - (1e-13*0.5)), (1e-13 + (1e-13*0.5)))

    # rxn2: OF + $H_out -> OF_Hb; vol*(rxn2_k1*OF*H_out - rxn2_k2*OF_Hb)
    p[0] = np.random.uniform(k_H_on_range[0], k_H_on_range[1]) # rxn2_k1
    p[1] = np.random.uniform(k_H_off_range[0], k_H_off_range[1]) # rxn2_k2

    # rxn3: OF_Sb -> OF + $S_out; vol*(rxn3_k1*OF_Sb - rxn3_k2*OF*S_out)
    p[2] = np.random.uniform(k_S_off_range[0], k_S_off_range[1]) # rxn3_k1
    p[3] = np.random.uniform(k_S_on_range[0], k_S_on_range[1]) # rxn3_k2

    # rxn4: OF_Hb -> IF_Hb; vol*(rxn4_k1*OF_Hb - rxn4_k2*IF_Hb)
    p[4] = np.random.uniform(k_conf_range[0], k_conf_range[1]) # rxn4_k1
    p[5] = np.random.uniform(k_conf_range[0], k_conf_range[1]) # rxn4_k2

    # rxn6: IF_Sb -> OF_Sb; vol*(rxn6_k1*IF_Sb - rxn6_k2*OF_Sb)
    p[6] = np.random.uniform(k_conf_range[0], k_conf_range[1]) # rxn6_k1
    p[7] = np.random.uniform(k_conf_range[0], k_conf_range[1]) # rxn6_k2

    # rxn11: IF_Hb + S_in -> IF_Hb_Sb; vol*(rxn11_k1*IF_Hb*S_in - rxn11_k2*IF_Hb_Sb)
    p[8] = np.random.uniform(k_S_on_range[0], k_S_on_range[1]) # rxn11_k1
    p[9] = np.random.uniform(k_S_off_range[0], k_S_off_range[1]) # rxn11_k2    

    # rxn12: IF_Hb_Sb -> IF_Sb + H_in; vol*(rxn12_k1*IF_Hb_Sb - rxn12_k2*IF_Sb*H_in)
    p[10] = np.random.uniform(k_H_off_range[0], k_H_off_range[1]) # rxn12_k1
    #p[11] = np.random.uniform(k_H_on_range[0], k_H_on_range[1]) # rxn12_k2    

    # experimental noise
    #p[12] = np.random.uniform(sigma_range[0], sigma_range[1]) # sigma
    p[11] = np.random.uniform(sigma_range[0], sigma_range[1]) # sigma
    
    return p


def set_reference_model_parameters(p, s=0):
    '''set parameter values for reference model'''
    # #s=0
    # k_conf_range = (-1-s,5)
    # k_H_on_range = (7-s,13)
    # k_H_off_range = (0-s,6)
    # k_S_on_range = (4-s,10)
    # k_S_off_range = (0-s,6)
    # #sigma_range = (np.log10(5e-15),np.log10(5e-12))
    # sigma_range = ((1e-13 - (1e-13*0.5)), (1e-13 + (1e-13*0.5)))

    # prior shift (testing)
    #s = [-1,1, -1, 1,-1]
    # s = [-0.5, 0.5, -0.5, 0.5, -0.5]
    s = [0, 0, 0, 0, 0,]
    k_conf_range = (-1-s[0],5-s[0])
    k_H_on_range = (7-s[1],13-s[1])
    k_H_off_range = (0-s[2],6-s[2])
    k_S_on_range = (4-s[3],10-s[3])
    k_S_off_range = (0-s[4],6-s[4])
    sigma_range = ((1e-13 - (1e-13*0.5)), (1e-13 + (1e-13*0.5)))


    k_H_on = np.log10(1e10)
    k_H_off = np.log10(1e3)
    k_S_on = np.log10(1e7)
    k_S_off = np.log10(1e3)
    k_conf = np.log10(1e2)
    #sigma = np.log10(1e-13)
    sigma = 1e-13
  

    # set all parameters to minimum value

    # rxn2: OF + $H_out -> OF_Hb; vol*(rxn2_k1*OF*H_out - rxn2_k2*OF_Hb)
    p[0] = k_H_on_range[0] # rxn2_k1
    p[1] = k_H_off_range[0]  # rxn2_k2

    # rxn3: OF_Sb -> OF + $S_out; vol*(rxn3_k1*OF_Sb - rxn3_k2*OF*S_out)
    p[2] = k_S_off_range[0]  # rxn3_k1
    p[3] = k_S_on_range[0]# rxn3_k2

    # rxn4: OF_Hb -> IF_Hb; vol*(rxn4_k1*OF_Hb - rxn4_k2*IF_Hb)
    p[4] = k_conf_range[0] # rxn4_k1
    p[5] = k_conf_range[0]  # rxn4_k2

    # rxn6: IF_Sb -> OF_Sb; vol*(rxn6_k1*IF_Sb - rxn6_k2*OF_Sb)
    p[6] = k_conf_range[0]  # rxn6_k1
    p[7] = k_conf_range[0]  # rxn6_k2

    # rxn11: IF_Hb + S_in -> IF_Hb_Sb; vol*(rxn11_k1*IF_Hb*S_in - rxn11_k2*IF_Hb_Sb)
    p[8] = k_S_on_range[0] # rxn11_k1
    p[9] = k_S_off_range[0]   # rxn11_k2    

    # rxn12: IF_Hb_Sb -> IF_Sb + H_in; vol*(rxn12_k1*IF_Hb_Sb - rxn12_k2*IF_Sb*H_in)
    p[10] = k_H_off_range[0] # rxn12_k1
    #p[11] = k_H_on_range[0] # rxn12_k2    

    # experimental noise
    #p[12] = sigma_range[0] # sigma
    p[11] = sigma_range[0] # sigma

    # update selected parameters reference values 
    p[0] = k_H_on
    p[1] = k_H_off
    p[2] = k_S_off
    p[3] = k_S_on
    p[4] = k_conf
    p[5] = k_conf
    p[6] = k_conf
    p[7] = k_conf
    p[8] = k_S_on
    p[9] = k_S_off
    p[10] = k_H_off
    p[11] = sigma
    
    return p


def check_prior(p, s=0):
    '''set parameter values for reference model'''
    # s = 0
    # k_conf_range = (-1-s,5)
    # k_H_on_range = (7-s,13)
    # k_H_off_range = (0-s,6)
    # k_S_on_range = (4-s,10)
    # k_S_off_range = (0-s,6)
    # # sigma_range = (np.log10(5e-14,5e-13))
    # sigma_range = ((1e-13 - (1e-13*0.5)), (1e-13 + (1e-13*0.5)))

    # prior shift (testing)
    # s = [-1,1, -1, 1,-1]
    # s = [-0.5, 0.5, -0.5, 0.5, -0.5]
    s = [0, 0, 0, 0, 0,]
    k_conf_range = (-1-s[0],5-s[0])
    k_H_on_range = (7-s[1],13-s[1])
    k_H_off_range = (0-s[2],6-s[2])
    k_S_on_range = (4-s[3],10-s[3])
    k_S_off_range = (0-s[4],6-s[4])
    sigma_range = ((1e-13 - (1e-13*0.5)), (1e-13 + (1e-13*0.5)))

    prior_dict = {}
    prior_dict['k_conf'] = [k_conf_range, [4,5,6,7]]
    prior_dict['k_H_on'] = [k_H_on_range, [0]]
    # prior_dict['k_H_on'] = [k_H_on_range, [0,11]]
    prior_dict['k_H_off'] = [k_H_off_range, [1,10]]
    prior_dict['k_S_on'] = [k_S_on_range, [3,8]]
    prior_dict['k_S_off'] = [k_S_off_range, [2,9]]
    # prior_dict['sigma'] = [sigma_range, [12]]
    prior_dict['sigma'] = [sigma_range, [11]]

    for key in prior_dict:
        tmp_range = prior_dict[key][0]
        tmp_idx_list = prior_dict[key][1]
        #print(f'{key}: range ({tmp_range[0]},{tmp_range[1]}), idx {tmp_idx_list}')
        for idx in tmp_idx_list:
            #print(f'  p[{idx}]: {tmp_range[0]} <= {p[idx]} <= {tmp_range[1]} ?')
            if not tmp_range[0] <= p[idx] <= tmp_range[1]:
                #print('  !ALERT: OUT OF RANGE! returning FALSE')
                return False
        #print('\n')
    #print('ALL IN RANGE! returning TRUE')
    return True

def energy_to_rate(p):
    '''convert parameter set in energies to rate constants'''
    e_bar = {}
    e_state = {}


# ##### TESTING

# params = {'legend.fontsize': 'xx-large',
#           'figure.figsize': (12, 10),
#          'axes.labelsize': 'xx-large',
#          'axes.titlesize':'xx-large',
#          'xtick.labelsize':'xx-large',
#          'ytick.labelsize':'xx-large'}
# plt.rcParams.update(params)

# ####
# sigma_ref = 1e-13
# k_H_on = np.log10(1e10)
# k_H_off = np.log10(1e3)
# k_S_on = np.log10(1e7)
# k_S_off = np.log10(1e3)
# k_conf = np.log10(1e2)

# p_synth = np.zeros(12)
# p_synth[0] = k_H_on
# p_synth[1] = k_H_off
# p_synth[2] = k_S_off
# p_synth[3] = k_S_on
# p_synth[4] = k_conf
# p_synth[5] = k_conf
# p_synth[6] = k_conf
# p_synth[7] = k_conf
# p_synth[8] = k_S_on
# p_synth[9] = k_S_off
# p_synth[10] = k_H_off
# #p_synth[11] = k_H_on
# #p_synth[12] = sigma_ref
# p_synth[11] = sigma_ref

# m = init_model(p_synth)



# y_ref = simulate_model(p_synth,m)
# datafile = '/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/scripts/transporter_int_2exp_2stage_half_data.csv'

# y_obs = np.loadtxt(f'{datafile}', delimiter=',', skiprows=1, usecols=1).tolist()  # load data from file

# import numpy as np
# fname = '/Users/georgeau/Desktop/research_data/local_macbook/intermediate_transporter/20220127_180709_intermediate_transporter/DEBUG_intermediate_transporter_20220127_180709_data.csv'
# n_cols = 12
# col_list = [i+1 for i in range(n_cols)]  # only keep columns 1...n_p-1 
# D = np.genfromtxt(fname, delimiter=',', skip_header=1,usecols=col_list)  
# inds = np.random.randint(len(D), size=100)

# for i, ind in enumerate(inds):
#     sample = D[ind]
  
#     y_pred_i = simulate_model(sample,m)
#     if i == len(inds)-1:
#         plt.plot(y_pred_i, alpha=0.5, color='black', label='predicted')
#     else:
#         plt.plot(y_pred_i, alpha=0.1, color='black')
# plt.plot(y_obs, 'o', alpha=0.5, label='observed')
# plt.ylabel('ion flux [M/s]')
# plt.xlabel('time [s]')

# locs, labels = plt.xticks()
# print(locs)
# print(labels)

# plt.xticks(locs[1:-1], locs[1:-1]*0.04)

# plt.legend()
# plt.tight_layout()
# plt.title(r'Comparing predicted and observed flux traces ')
# plt.savefig('output.png')

# ####



### intialization
seed = 1234
np.random.seed(seed)
start_time = datetime.now()
time_str = time.strftime("%Y%m%d_%H%M%S") 
filename=f'intermediate_transporter_{time_str}'
new_dir = pathlib.Path('/Users/georgeau/Desktop/research_data/local_macbook/intermediate_transporter/', f'{time_str}_intermediate_transporter')
new_dir.mkdir(parents=True, exist_ok=True)

sigma_ref = 1e-13
k_H_on = np.log10(1e10)
k_H_off = np.log10(1e3)
k_S_on = np.log10(1e7)
k_S_off = np.log10(1e3)
k_conf = np.log10(1e2)

p_synth = np.zeros(12)
p_synth[0] = k_H_on
p_synth[1] = k_H_off
p_synth[2] = k_S_off
p_synth[3] = k_S_on
p_synth[4] = k_conf
p_synth[5] = k_conf
p_synth[6] = k_conf
p_synth[7] = k_conf
p_synth[8] = k_S_on
p_synth[9] = k_S_off
p_synth[10] = k_H_off
#p_synth[11] = k_H_on
#p_synth[12] = sigma_ref
p_synth[11] = sigma_ref

m = init_model(p_synth)
y_ref = simulate_model(p_synth,m)

labels = [
    'rxn2_k1',
    'rxn2_k2',
    'rxn3_k1',
    'rxn3_k2',
    'rxn4_k1',
    'rxn4_k2',
    'rxn6_k1',
    'rxn6_k2',
    'rxn11_k1',
    'rxn11_k2',
    'rxn12_k1',
    'sigma'
]


#datafile = '/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/scripts/transporter_int_2exp_2stage_all_data.csv'
datafile = '/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/scripts/transporter_int_2exp_2stage_half_data.csv'

y_obs = np.loadtxt(f'{datafile}', delimiter=',', skiprows=1, usecols=1).tolist()  # load data from file

max_logl_synth = log_likelihood(p_synth, y_obs, m)
print(max_logl_synth)
print(p_synth)


# p_ref = [0]*25

n_walkers = 50
n_steps = int(1e4)
n_burn = int(0.1*n_steps)
n_dim = 12
n_temps = 4
move_list = []

# testing

if use_pt_sampler==True:
    p0_list = []
    for j in range(n_temps):
        pos_list = []
        for i in range(n_walkers):
            p0_tmp = randomize_model_parameters(np.zeros(n_dim))
            pos_list.append(p0_tmp)
        p0_list.append(pos_list)
    p0 = np.asarray(p0_list)
    assert(np.shape(p0) == (n_temps,n_walkers,n_dim))

    sampler=mc.PTSampler(n_temps, n_walkers, n_dim, log_likelihood, log_prior, loglargs=[y_obs, m] )


    i=0
    for p, lnprob, lnlike in sampler.sample(p0, iterations=n_burn):
        print(f'{i+1}/{n_burn}')
        i+=1
    
    pt_samples_burn = sampler.flatchain[0,:,:]
    logl_burn = sampler.lnlikelihood[0,:,:].reshape((-1))
    samples_df_burn = pd.DataFrame(pt_samples_burn, columns=labels)
    samples_df_burn['logl'] = logl_burn
    samples_df_burn.to_csv(new_dir/f'{filename}_data_burn.csv')

    sampler.reset()
    i=0
    for p, lnprob, lnlike in sampler.sample(p, lnprob0=lnprob,
                                            lnlike0=lnlike,
                                            iterations=n_steps):
        print(f'{i+1}/{n_steps}')
        i+=1
        pass
    assert sampler.chain.shape == (n_temps, n_walkers, n_steps, n_dim)
else:
    move_list=[
        (mc.moves.DESnookerMove(), 0.5),
        (mc.moves.StretchMove(), 0.5)
    ]
    p0_list = []
    for i in range(n_walkers):
        p0_tmp = randomize_model_parameters(np.zeros(n_dim))
        p0_list.append(p0_tmp)
    p0 = np.asarray(p0_list)
    print(np.shape(p0))
    assert(np.shape(p0) == (n_walkers,n_dim))
    ### sampling
    sampler = mc.EnsembleSampler(n_walkers, n_dim, log_probability, args=[y_obs,m], moves=move_list)
    sampler.run_mcmc(p0, int(n_steps+n_burn), progress=True)
end_time = datetime.now()


#################################################################

### analysis
# labels = [
#     'rxn2_k1',
#     'rxn2_k2',
#     'rxn3_k1',
#     'rxn3_k2',
#     'rxn4_k1',
#     'rxn4_k2',
#     'rxn6_k1',
#     'rxn6_k2',
#     'rxn11_k1',
#     'rxn11_k2',
#     'rxn12_k1',
#     'rxn12_k2',
#     'sigma'
# ]
labels = [
    'rxn2_k1',
    'rxn2_k2',
    'rxn3_k1',
    'rxn3_k2',
    'rxn4_k1',
    'rxn4_k2',
    'rxn6_k1',
    'rxn6_k2',
    'rxn11_k1',
    'rxn11_k2',
    'rxn12_k1',
    'sigma'
]

with open(new_dir/f'{time_str}_emcee_transporter_log.txt', 'a') as f:
        f.write(f'{time_str}_emcee_transporter_log.txt\n\n')
        f.write(f'timestamp:{time_str}\n')
        f.write(f'n walkers:{n_walkers}\nn steps/walker:{n_steps}\nn temps:{n_temps}\nusing PT sampler:{use_pt_sampler}\nmoves:{move_list}\n' )  
        f.write(f'datafile:{datafile}\nparameters:{labels}\nparameter reference values:{p_synth}\n')
        f.write(f'less data: {less_data}\ndata amount: {data_amount}\nn data points: {len(y_obs)}\n3 experiments: {three_exp}\n')
        f.write(f'max logl (synth): {max_logl_synth}\n')


s=0
sigma_ref = 1e-13
k_H_on = np.log10(1e10)
k_H_off = np.log10(1e3)
k_S_on = np.log10(1e7)
k_S_off = np.log10(1e3)
k_conf = np.log10(1e2)


# k_conf_range = (-1-s,5)
# k_H_on_range = (7-s,13)
# k_H_off_range = (0-s,6)
# k_S_on_range = (4-s,10)
# k_S_off_range = (0-s,6)
# # sigma_range = (np.log10(5e-14,5e-13))
# sigma_range = ((1e-13 - (1e-13*0.5)), (1e-13 + (1e-13*0.5)))

# prior shift (testing)
# s = [-1,1, -1, 1,-1]
# s = [-0.5, 0.5, -0.5, 0.5, -0.5]
s = [0, 0, 0, 0, 0,]
k_conf_range = (-1-s[0],5-s[0])
k_H_on_range = (7-s[1],13-s[1])
k_H_off_range = (0-s[2],6-s[2])
k_S_on_range = (4-s[3],10-s[3])
k_S_off_range = (0-s[4],6-s[4])
sigma_range = ((1e-13 - (1e-13*0.5)), (1e-13 + (1e-13*0.5)))

### boundary ranges
bounds=[0]*n_dim

# rxn2: OF + $H_out -> OF_Hb; vol*(rxn2_k1*OF*H_out - rxn2_k2*OF_Hb)
bounds[0] = k_H_on_range # rxn2_k1
bounds[1] = k_H_off_range  # rxn2_k2

# rxn3: OF_Sb -> OF + $S_out; vol*(rxn3_k1*OF_Sb - rxn3_k2*OF*S_out)
bounds[2] = k_S_off_range  # rxn3_k1
bounds[3] = k_S_on_range # rxn3_k2

# rxn4: OF_Hb -> IF_Hb; vol*(rxn4_k1*OF_Hb - rxn4_k2*IF_Hb)
bounds[4] = k_conf_range # rxn4_k1
bounds[5] = k_conf_range # rxn4_k2

# rxn6: IF_Sb -> OF_Sb; vol*(rxn6_k1*IF_Sb - rxn6_k2*OF_Sb)
bounds[6] = k_conf_range # rxn6_k1
bounds[7] = k_conf_range # rxn6_k2

# rxn11: IF_Hb + S_in -> IF_Hb_Sb; vol*(rxn11_k1*IF_Hb*S_in - rxn11_k2*IF_Hb_Sb)
bounds[8] = k_S_on_range  # rxn11_k1
bounds[9] = k_S_off_range   # rxn11_k2    

# rxn12: IF_Hb_Sb -> IF_Sb + H_in; vol*(rxn12_k1*IF_Hb_Sb - rxn12_k2*IF_Sb*H_in)
bounds[10] = k_H_off_range  # rxn12_k1
# bounds[11] = k_H_on_range  # rxn12_k2    

# experimental noise
# bounds[12] = sigma_range # sigma  
bounds[11] = sigma_range # sigma  
print(bounds)


### reference values
p_ref = np.zeros(n_dim)


p_ref[0] = k_H_on
p_ref[1] = k_H_off
p_ref[2] = k_S_off
p_ref[3] = k_S_on
p_ref[4] = k_conf
p_ref[5] = k_conf
p_ref[6] = k_conf
p_ref[7] = k_conf
p_ref[8] = k_S_on
p_ref[9] = k_S_off
p_ref[10] = k_H_off
# p_ref[11] = k_H_on
# p_ref[12] = sigma_ref
p_ref[11] = sigma_ref
print(p_ref)




# mcmc trajectory (before burn in)


if use_pt_sampler == True:

    samples = np.transpose(sampler.chain[0,:,:,:])
    print(np.shape(samples))
    print(np.size(samples))
    fig, axes = plt.subplots(n_dim, figsize=(20, 15), sharex=True)
    for i in range(n_dim):
        ax = axes[i]
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.set_ylim(bounds[i])
        
        for j in range(n_walkers):
            ax.plot(samples[i][j*n_steps:((j+1)*n_steps)-1], "k", alpha=0.1)
            
        ax.axhline(p_ref[i], linestyle='--', color='red', alpha=0.7)
    axes[-1].set_xlabel("step number");
    plt.savefig(new_dir/f'{filename}_traces.png')

else: 
    samples = sampler.get_chain()
    fig, axes = plt.subplots(n_dim, figsize=(20, 15), sharex=True)
    for i in range(n_dim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    axes[-1].set_xlabel("step number");
    plt.savefig(new_dir/f'{filename}_traces.png')



# autocorrelation time (for measuring sampling performance)
# note: this can give an error if the run length isn't 50x the autocorrelation
try:
    tau = sampler.get_autocorr_time()
    print(tau)
except:
    print('error in tau calculation - check chain convergence')


# corner plot (1d and 2d histograms)
if use_pt_sampler == True:
    pt_samples = sampler.flatchain[0,:,:]
    flat_samples = pt_samples
    logl = sampler.lnlikelihood[0,:,:].reshape((-1))
else:
    flat_samples = sampler.get_chain(discard=n_burn, flat=True)
    logl = sampler.get_log_prob(discard=n_burn,  flat=True)

logl_max_sample = np.max(logl)
with open(new_dir/f'{time_str}_emcee_transporter_log.txt', 'a') as f:
        f.write(f'max logl sampled: {logl_max_sample}\n')
        f.write(f'{filename}_data.csv\n')
        f.write('wall clock: {}'.format(end_time - start_time))


print(np.size(flat_samples))
print(np.shape(flat_samples))
print(np.max(logl))
print(np.min(logl))
print(f'any nan logl? {np.isnan(logl).any()}')
print(f'any nan parameters? {np.isnan(flat_samples).any()}')
samples_df = pd.DataFrame(flat_samples, columns=labels)
samples_df['logl'] = logl
samples_df.to_csv(new_dir/f'{filename}_data.csv')

#p_ref = np.append(k_ref, np.log10(1e-13))

theta_true = p_ref

fig = corner.corner(
        flat_samples, labels=labels, truths=theta_true
    )
plt.savefig(new_dir/f'{filename}_2dcorr.png')


# plot y_predicted and y_observed
inds = np.random.randint(len(flat_samples), size=100)
plt.figure(figsize = (15, 10))
for ind in inds:
    sample = flat_samples[ind]
    y_pred_i = simulate_model(sample,m)
    plt.plot(y_pred_i, alpha=0.1, color='black')
plt.title('y_pred and y_obs')
plt.ylabel('y')
plt.xlabel('x')
plt.plot(y_obs, ls='None', color='orange', marker='o',label="observed", alpha=0.6)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig(new_dir/f'{filename}_example_plots.png')

fig, axes = plt.subplots(3, 4, figsize=(10,10))
ax = axes.flatten()

d_dict = {}

ref_idx_list = [0,1,2,3,4,5,6,7,8,9,10,11,12]
flat_samples_T = np.transpose(flat_samples)
for i, lbl in enumerate(labels):
    p_data = flat_samples_T[i]
    d_dict[lbl] = p_data
    
    if i in ref_idx_list:
        ax[i].hist(p_data, alpha=0.7, bins=100, range=bounds[i], color='green', density=True)
    else:
        ax[i].hist(p_data, alpha=0.7, bins=100, range=bounds[i], color='red', density=True)
    ax[i].axvline(x=p_ref[i], ymin=0, ymax=1, color='black', ls='--')
    ax[i].set_title(f'{labels[i]}')
plt.suptitle('1d marginal posterior distributions (log10 rate constants)')
plt.tight_layout()
plt.savefig(new_dir/f'{filename}_1d_post.png')