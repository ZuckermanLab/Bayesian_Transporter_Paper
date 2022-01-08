import tellurium as te
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import emcee as mc
import corner
import pandas as pd
import time
import pathlib 


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
            E1: at (time >= 5.0): H_out = H_out_activation;
            E2: at (time >= 10.0): H_out = 1e-7;

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

            // Rate constant initializations:
            rxn1_k1 = {theta[0]};
            rxn1_k2 = {theta[1]};
            rxn2_k1 = {theta[2]};
            rxn2_k2 = {theta[3]};
            rxn3_k1 = {theta[4]};
            rxn3_k2 = {theta[5]};
            rxn4_k1 = {theta[6]};
            rxn4_k2 = {theta[7]};
            rxn5_k1 = {theta[8]};
            rxn5_k2 = {theta[9]};
            rxn6_k1 = {theta[10]};
            rxn6_k2 = {theta[11]};
            rxn7_k1 = {theta[12]};
            rxn7_k2 = {theta[13]};
            rxn8_k1 = {theta[14]};
            rxn8_k2 = {theta[15]};
            rxn9_k1 = {theta[16]};
            rxn9_k2 = {theta[17]};
            rxn10_k1 = {theta[18]};
            rxn10_k2 = {theta[19]};
            rxn11_k1 = {theta[20]};
            rxn11_k2 = {theta[21]};
            rxn12_k1 = {theta[22]};
            rxn12_k2 = {theta[23]};


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

    # reset z to initial
    z.resetToOrigin()

    #update pH
    z.H_out_activation = 5e-8

    # update rate constants
    z.rxn1_k1 = 10**p[0]
    z.rxn1_k2 = 10**p[1]
    z.rxn2_k1 = 10**p[2]
    z.rxn2_k2 = 10**p[3]
    z.rxn3_k1 = 10**p[4]
    z.rxn3_k2 = 10**p[5]
    z.rxn4_k1 = 10**p[6]
    z.rxn4_k2 = 10**p[7]
    z.rxn5_k1 = 10**p[8]
    z.rxn5_k2 = 10**p[9]
    z.rxn6_k1 = 10**p[10]
    z.rxn6_k2 = 10**p[11]
    z.rxn7_k1 = 10**p[12]
    z.rxn7_k2 = 10**p[13]
    z.rxn8_k1 = 10**p[14]
    z.rxn8_k2 = 10**p[15]
    z.rxn9_k1 = 10**p[16]
    z.rxn9_k2 = 10**p[17]
    z.rxn10_k1 = 10**p[18]
    z.rxn10_k2 = 10**p[19]
    z.rxn11_k1 = 10**p[20]
    z.rxn11_k2 = 10**p[21]
    z.rxn12_k1 = 10**p[22]
    z.rxn12_k2 = 10**p[23]

    
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

        y_pred_1 = np.hstack([s2,s3])
    except:
        return np.zeros(500)

    # reset z to initial
    z.resetToOrigin()

    #update pH
    z.H_out_activation = 5e-7

    # update rate constants
    z.rxn1_k1 = 10**p[0]
    z.rxn1_k2 = 10**p[1]
    z.rxn2_k1 = 10**p[2]
    z.rxn2_k2 = 10**p[3]
    z.rxn3_k1 = 10**p[4]
    z.rxn3_k2 = 10**p[5]
    z.rxn4_k1 = 10**p[6]
    z.rxn4_k2 = 10**p[7]
    z.rxn5_k1 = 10**p[8]
    z.rxn5_k2 = 10**p[9]
    z.rxn6_k1 = 10**p[10]
    z.rxn6_k2 = 10**p[11]
    z.rxn7_k1 = 10**p[12]
    z.rxn7_k2 = 10**p[13]
    z.rxn8_k1 = 10**p[14]
    z.rxn8_k2 = 10**p[15]
    z.rxn9_k1 = 10**p[16]
    z.rxn9_k2 = 10**p[17]
    z.rxn10_k1 = 10**p[18]
    z.rxn10_k2 = 10**p[19]
    z.rxn11_k1 = 10**p[20]
    z.rxn11_k2 = 10**p[21]
    z.rxn12_k1 = 10**p[22]
    z.rxn12_k2 = 10**p[23]

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
        y_pred_2 = np.hstack([s2_2,s3_2])
        y_pred = np.hstack([y_pred_1, y_pred_2])
    except:
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


def randomize_model_parameters(p):
    '''randomize initial parameter values to be within accepted range for each parameter type'''
    s=4
    k_conf_range = (-1-s,5)
    k_H_on_range = (7-s,13)
    k_H_off_range = (0-s,6)
    k_S_on_range = (4-s,10)
    k_S_off_range = (0-s,6)
    # sigma_range = (np.log10(5e-14),np.log10(5e-13))
    sigma_range = ((1e-13 - (1e-13*0.5)), (1e-13 + (1e-13*0.5)))

    # rxn1: IF -> OF; vol*(rxn1_k1*IF - rxn1_k2*OF)
    p[0] = np.random.uniform(k_conf_range[0], k_conf_range[1]) # rxn1_k1
    p[1] = np.random.uniform(k_conf_range[0], k_conf_range[1]) # rxn1_k2

    # rxn2: OF + $H_out -> OF_Hb; vol*(rxn2_k1*OF*H_out - rxn2_k2*OF_Hb)
    p[2] = np.random.uniform(k_H_on_range[0], k_H_on_range[1]) # rxn2_k1
    p[3] = np.random.uniform(k_H_off_range[0], k_H_off_range[1]) # rxn2_k2

    # rxn3: OF_Sb -> OF + $S_out; vol*(rxn3_k1*OF_Sb - rxn3_k2*OF*S_out)
    p[4] = np.random.uniform(k_S_off_range[0], k_S_off_range[1]) # rxn3_k1
    p[5] = np.random.uniform(k_S_on_range[0], k_S_on_range[1]) # rxn3_k2

    # rxn4: OF_Hb -> IF_Hb; vol*(rxn4_k1*OF_Hb - rxn4_k2*IF_Hb)
    p[6] = np.random.uniform(k_conf_range[0], k_conf_range[1]) # rxn4_k1
    p[7] = np.random.uniform(k_conf_range[0], k_conf_range[1]) # rxn4_k2

    # rxn5: OF_Hb_Sb -> OF_Hb + $S_out; vol*(rxn5_k1*OF_Hb_Sb - rxn5_k2*OF_Hb*S_out)
    p[8] = np.random.uniform(k_S_off_range[0], k_S_off_range[1]) # rxn5_k1
    p[9] = np.random.uniform(k_S_on_range[0], k_S_on_range[1]) # rxn5_k2

    # rxn6: IF_Sb -> OF_Sb; vol*(rxn6_k1*IF_Sb - rxn6_k2*OF_Sb)
    p[10] = np.random.uniform(k_conf_range[0], k_conf_range[1]) # rxn6_k1
    p[11] = np.random.uniform(k_conf_range[0], k_conf_range[1]) # rxn6_k2

    # rxn7: OF_Sb + $H_out -> OF_Hb_Sb; vol*(rxn7_k1*OF_Sb*H_out - rxn7_k2*OF_Hb_Sb)
    p[12] = np.random.uniform(k_H_on_range[0], k_H_on_range[1]) # rxn7_k1
    p[13] = np.random.uniform(k_H_off_range[0], k_H_off_range[1]) # rxn7_k2

    # rxn8: OF_Hb_Sb -> IF_Hb_Sb; vol*(rxn8_k1*OF_Hb_Sb - rxn8_k2*IF_Hb_Sb)
    p[14] = np.random.uniform(k_conf_range[0], k_conf_range[1]) # rxn8_k1 
    p[15] = np.random.uniform(k_conf_range[0], k_conf_range[1]) # rxn8_k2

    # rxn9: IF_Hb -> IF + H_in; vol*(rxn9_k1*IF_Hb - rxn9_k2*IF*H_in)
    p[16] = np.random.uniform(k_H_off_range[0], k_H_off_range[1]) # rxn9_k1
    p[17] = np.random.uniform(k_H_on_range[0], k_H_on_range[1]) # rxn9_k2   

    # rxn10: IF + S_in -> IF_Sb; vol*(rxn10_k1*IF*S_in - rxn10_k2*IF_Sb)
    p[18] = np.random.uniform(k_S_on_range[0], k_S_on_range[1]) # rxn10_k1
    p[19] = np.random.uniform(k_S_off_range[0], k_S_off_range[1]) # rxn10_k2    

    # rxn11: IF_Hb + S_in -> IF_Hb_Sb; vol*(rxn11_k1*IF_Hb*S_in - rxn11_k2*IF_Hb_Sb)
    p[20] = np.random.uniform(k_S_on_range[0], k_S_on_range[1]) # rxn11_k1
    p[21] = np.random.uniform(k_S_off_range[0], k_S_off_range[1]) # rxn11_k2    

    # rxn12: IF_Hb_Sb -> IF_Sb + H_in; vol*(rxn12_k1*IF_Hb_Sb - rxn12_k2*IF_Sb*H_in)
    p[22] = np.random.uniform(k_H_off_range[0], k_H_off_range[1]) # rxn12_k1
    p[23] = np.random.uniform(k_H_on_range[0], k_H_on_range[1]) # rxn12_k2    

    # experimental noise
    p[24] = np.random.uniform(sigma_range[0], sigma_range[1]) # sigma
    
    return p


def set_reference_model_parameters(p):
    '''set parameter values for reference model'''
    s=4
    k_conf_range = (-1-s,5)
    k_H_on_range = (7-s,13)
    k_H_off_range = (0-s,6)
    k_S_on_range = (4-s,10)
    k_S_off_range = (0-s,6)
    #sigma_range = (np.log10(5e-15),np.log10(5e-12))
    sigma_range = ((1e-13 - (1e-13*0.5)), (1e-13 + (1e-13*0.5)))


    k_H_on = np.log10(1e10)
    k_H_off = np.log10(1e3)
    k_S_on = np.log10(1e7)
    k_S_off = np.log10(1e3)
    k_conf = np.log10(1e2)
    #sigma = np.log10(1e-13)
    sigma = 1e-13
  

    # set all parameters to minimum value
    # rxn1: IF -> OF; vol*(rxn1_k1*IF - rxn1_k2*OF)
    p[0] = k_conf_range[0]  # rxn1_k1
    p[1] = k_conf_range[0]  # rxn1_k2

    # rxn2: OF + $H_out -> OF_Hb; vol*(rxn2_k1*OF*H_out - rxn2_k2*OF_Hb)
    p[2] = k_H_on_range[0] # rxn2_k1
    p[3] = k_H_off_range[0]  # rxn2_k2

    # rxn3: OF_Sb -> OF + $S_out; vol*(rxn3_k1*OF_Sb - rxn3_k2*OF*S_out)
    p[4] = k_S_off_range[0]  # rxn3_k1
    p[5] = k_S_on_range[0]# rxn3_k2

    # rxn4: OF_Hb -> IF_Hb; vol*(rxn4_k1*OF_Hb - rxn4_k2*IF_Hb)
    p[6] = k_conf_range[0] # rxn4_k1
    p[7] = k_conf_range[0]  # rxn4_k2

    # rxn5: OF_Hb_Sb -> OF_Hb + $S_out; vol*(rxn5_k1*OF_Hb_Sb - rxn5_k2*OF_Hb*S_out)
    p[8] = k_S_off_range[0]  # rxn5_k1
    p[9] = k_S_on_range[0]  # rxn5_k2

    # rxn6: IF_Sb -> OF_Sb; vol*(rxn6_k1*IF_Sb - rxn6_k2*OF_Sb)
    p[10] = k_conf_range[0]  # rxn6_k1
    p[11] = k_conf_range[0]  # rxn6_k2

    # rxn7: OF_Sb + $H_out -> OF_Hb_Sb; vol*(rxn7_k1*OF_Sb*H_out - rxn7_k2*OF_Hb_Sb)
    p[12] = k_H_on_range[0]  # rxn7_k1
    p[13] = k_H_off_range[0] # rxn7_k2

    # rxn8: OF_Hb_Sb -> IF_Hb_Sb; vol*(rxn8_k1*OF_Hb_Sb - rxn8_k2*IF_Hb_Sb)
    p[14] = k_conf_range[0] # rxn8_k1 
    p[15] = k_conf_range[0] # rxn8_k2

    # rxn9: IF_Hb -> IF + H_in; vol*(rxn9_k1*IF_Hb - rxn9_k2*IF*H_in)
    p[16] = k_H_off_range[0]  # rxn9_k1
    p[17] = k_H_on_range[0]   # rxn9_k2   

    # rxn10: IF + S_in -> IF_Sb; vol*(rxn10_k1*IF*S_in - rxn10_k2*IF_Sb)
    p[18] = k_S_on_range[0]   # rxn10_k1
    p[19] = k_S_off_range[0]  # rxn10_k2    

    # rxn11: IF_Hb + S_in -> IF_Hb_Sb; vol*(rxn11_k1*IF_Hb*S_in - rxn11_k2*IF_Hb_Sb)
    p[20] = k_S_on_range[0] # rxn11_k1
    p[21] = k_S_off_range[0]   # rxn11_k2    

    # rxn12: IF_Hb_Sb -> IF_Sb + H_in; vol*(rxn12_k1*IF_Hb_Sb - rxn12_k2*IF_Sb*H_in)
    p[22] = k_H_off_range[0] # rxn12_k1
    p[23] = k_H_on_range[0] # rxn12_k2    

    # experimental noise
    p[24] = sigma_range[0] # sigma

    # update selected parameters reference values 
    p[2] = k_H_on
    p[3] = k_H_off
    p[4] = k_S_off
    p[5] = k_S_on
    p[6] = k_conf
    p[7] = k_conf
    p[10] = k_conf
    p[11] = k_conf
    p[20] = k_S_on
    p[21] = k_S_off
    p[22] = k_H_off
    p[23] = k_H_on
    p[24] = sigma
    
    return p


def check_prior(p):
    '''set parameter values for reference model'''
    s = 4
    k_conf_range = (-1-s,5)
    k_H_on_range = (7-s,13)
    k_H_off_range = (0-s,6)
    k_S_on_range = (4-s,10)
    k_S_off_range = (0-s,6)
    # sigma_range = (np.log10(5e-14,5e-13))
    sigma_range = ((1e-13 - (1e-13*0.5)), (1e-13 + (1e-13*0.5)))


    prior_dict = {}
    prior_dict['k_conf'] = [k_conf_range, [0,1,6,7,10,11,14,15]]
    prior_dict['k_H_on'] = [k_H_on_range, [2,12,17,23]]
    prior_dict['k_H_off'] = [k_H_off_range, [3,13,16,22]]
    prior_dict['k_S_on'] = [k_S_on_range, [5,9,18,20]]
    prior_dict['k_S_off'] = [k_S_off_range, [4,8,19,21]]
    prior_dict['sigma'] = [sigma_range, [24]]

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


##### TESTING

### intialization
time_str = time.strftime("%Y%m%d_%H%M%S") 
filename=f'full_transporter_{time_str}'
new_dir = pathlib.Path('/Users/georgeau/Desktop/research_data/local_macbook/full_transporter/', f'{time_str}_full_transporter')
new_dir.mkdir(parents=True, exist_ok=True)

sigma_ref = 1e-13
k_H_on = np.log10(1e10)
k_H_off = np.log10(1e3)
k_S_on = np.log10(1e7)
k_S_off = np.log10(1e3)
k_conf = np.log10(1e2)

p_synth = np.zeros(25)
p_synth = p_synth - 13
p_synth[2] = k_H_on
p_synth[3] = k_H_off
p_synth[4] = k_S_off
p_synth[5] = k_S_on
p_synth[6] = k_conf
p_synth[7] = k_conf
p_synth[10] = k_conf
p_synth[11] = k_conf
p_synth[20] = k_S_on
p_synth[21] = k_S_off
p_synth[22] = k_H_off
p_synth[23] = k_H_on
p_synth[24] = sigma_ref
print(p_synth)

m = init_model(p_synth)
y_ref = simulate_model(p_synth,m)
datafile = '/Users/georgeau/Desktop/GitHub/august/model_identification/affine_MCMC_PT/emcee_full_transporter_data_2stage_2ph_v2.csv'

y_obs = np.loadtxt(f'{datafile}', delimiter=',', skiprows=1, usecols=1).tolist()  # load data from file

# print(log_likelihood(p_synth,y_obs,m))
# p_ref = [0]*25
# print(log_likelihood(set_reference_model_parameters(p_ref), y_obs, m))

# exit()
seed = 1234
np.random.seed(seed)
n_walkers = 100
n_steps = 1e5
n_burn = int(0.1*n_steps)
n_dim = 25

p0_list = []
for i in range(n_walkers):
    p0_tmp = randomize_model_parameters(np.zeros(25))
    p0_list.append(p0_tmp)
p0 = np.asarray(p0_list)
assert(np.shape(p0) == (n_walkers,n_dim))

### sampling
sampler = mc.EnsembleSampler(n_walkers, n_dim, log_probability, args=[y_obs,m], moves=mc.moves.KDEMove())
sampler.run_mcmc(p0, int(n_steps+n_burn), progress=True)

#################################################################

### analysis
labels = [
    'rxn1_k1',
    'rxn1_k2',
    'rxn2_k1',
    'rxn2_k2',
    'rxn3_k1',
    'rxn3_k2',
    'rxn4_k1',
    'rxn4_k2',
    'rxn5_k1',
    'rxn5_k2',
    'rxn6_k1',
    'rxn6_k2',
    'rxn7_k1',
    'rxn7_k2',
    'rxn8_k1',
    'rxn8_k2',
    'rxn9_k1',
    'rxn9_k2',
    'rxn10_k1',
    'rxn10_k2',
    'rxn11_k1',
    'rxn11_k2',
    'rxn12_k1',
    'rxn12_k2',
    'sigma'
]

s=4
sigma_ref = 1e-13
k_H_on = np.log10(1e10)
k_H_off = np.log10(1e3)
k_S_on = np.log10(1e7)
k_S_off = np.log10(1e3)
k_conf = np.log10(1e2)

k_conf_range = (-1-s,5)
k_H_on_range = (7-s,13)
k_H_off_range = (0-s,6)
k_S_on_range = (4-s,10)
k_S_off_range = (0-s,6)
# sigma_range = (np.log10(5e-14,5e-13))
sigma_range = ((1e-13 - (1e-13*0.5)), (1e-13 + (1e-13*0.5)))


### boundary ranges
bounds=[0]*25
bounds[0] = k_conf_range  # rxn1_k1
bounds[1] = k_conf_range  # rxn1_k2

# rxn2: OF + $H_out -> OF_Hb; vol*(rxn2_k1*OF*H_out - rxn2_k2*OF_Hb)
bounds[2] = k_H_on_range # rxn2_k1
bounds[3] = k_H_off_range  # rxn2_k2

# rxn3: OF_Sb -> OF + $S_out; vol*(rxn3_k1*OF_Sb - rxn3_k2*OF*S_out)
bounds[4] = k_S_off_range  # rxn3_k1
bounds[5] = k_S_on_range # rxn3_k2

# rxn4: OF_Hb -> IF_Hb; vol*(rxn4_k1*OF_Hb - rxn4_k2*IF_Hb)
bounds[6] = k_conf_range # rxn4_k1
bounds[7] = k_conf_range # rxn4_k2

# rxn5: OF_Hb_Sb -> OF_Hb + $S_out; vol*(rxn5_k1*OF_Hb_Sb - rxn5_k2*OF_Hb*S_out)
bounds[8] = k_S_off_range # rxn5_k1
bounds[9] = k_S_on_range # rxn5_k2

# rxn6: IF_Sb -> OF_Sb; vol*(rxn6_k1*IF_Sb - rxn6_k2*OF_Sb)
bounds[10] = k_conf_range # rxn6_k1
bounds[11] = k_conf_range # rxn6_k2

# rxn7: OF_Sb + $H_out -> OF_Hb_Sb; vol*(rxn7_k1*OF_Sb*H_out - rxn7_k2*OF_Hb_Sb)
bounds[12] = k_H_on_range # rxn7_k1
bounds[13] = k_H_off_range  # rxn7_k2

# rxn8: OF_Hb_Sb -> IF_Hb_Sb; vol*(rxn8_k1*OF_Hb_Sb - rxn8_k2*IF_Hb_Sb)
bounds[14] = k_conf_range # rxn8_k1 
bounds[15] = k_conf_range  # rxn8_k2

# rxn9: IF_Hb -> IF + H_in; vol*(rxn9_k1*IF_Hb - rxn9_k2*IF*H_in)
bounds[16] = k_H_off_range  # rxn9_k1
bounds[17] = k_H_on_range   # rxn9_k2   

# rxn10: IF + S_in -> IF_Sb; vol*(rxn10_k1*IF*S_in - rxn10_k2*IF_Sb)
bounds[18] = k_S_on_range   # rxn10_k1
bounds[19] = k_S_off_range   # rxn10_k2    

# rxn11: IF_Hb + S_in -> IF_Hb_Sb; vol*(rxn11_k1*IF_Hb*S_in - rxn11_k2*IF_Hb_Sb)
bounds[20] = k_S_on_range  # rxn11_k1
bounds[21] = k_S_off_range   # rxn11_k2    

# rxn12: IF_Hb_Sb -> IF_Sb + H_in; vol*(rxn12_k1*IF_Hb_Sb - rxn12_k2*IF_Sb*H_in)
bounds[22] = k_H_off_range  # rxn12_k1
bounds[23] = k_H_on_range  # rxn12_k2    

# experimental noise
bounds[24] = sigma_range # sigma  
print(bounds)

### reference values
p_ref = np.zeros(25)

p_ref[0] = k_conf_range[0]  # rxn1_k1
p_ref[1] = k_conf_range[0]  # rxn1_k2

# rxn2: OF + $H_out -> OF_Hb; vol*(rxn2_k1*OF*H_out - rxn2_k2*OF_Hb)
p_ref[2] = k_H_on_range[0] # rxn2_k1
p_ref[3] = k_H_off_range[0]  # rxn2_k2

# rxn3: OF_Sb -> OF + $S_out; vol*(rxn3_k1*OF_Sb - rxn3_k2*OF*S_out)
p_ref[4] = k_S_off_range[0]  # rxn3_k1
p_ref[5] = k_S_on_range[0] # rxn3_k2

# rxn4: OF_Hb -> IF_Hb; vol*(rxn4_k1*OF_Hb - rxn4_k2*IF_Hb)
p_ref[6] = k_conf_range[0] # rxn4_k1
p_ref[7] = k_conf_range[0] # rxn4_k2

# rxn5: OF_Hb_Sb -> OF_Hb + $S_out; vol*(rxn5_k1*OF_Hb_Sb - rxn5_k2*OF_Hb*S_out)
p_ref[8] = k_S_off_range[0] # rxn5_k1
p_ref[9] = k_S_on_range[0] # rxn5_k2

# rxn6: IF_Sb -> OF_Sb; vol*(rxn6_k1*IF_Sb - rxn6_k2*OF_Sb)
p_ref[10] = k_conf_range[0] # rxn6_k1
p_ref[11] = k_conf_range[0] # rxn6_k2

# rxn7: OF_Sb + $H_out -> OF_Hb_Sb; vol*(rxn7_k1*OF_Sb*H_out - rxn7_k2*OF_Hb_Sb)
p_ref[12] = k_H_on_range[0] # rxn7_k1
p_ref[13] = k_H_off_range[0]  # rxn7_k2

# rxn8: OF_Hb_Sb -> IF_Hb_Sb; vol*(rxn8_k1*OF_Hb_Sb - rxn8_k2*IF_Hb_Sb)
p_ref[14] = k_conf_range[0] # rxn8_k1 
p_ref[15] = k_conf_range[0]  # rxn8_k2

# rxn9: IF_Hb -> IF + H_in; vol*(rxn9_k1*IF_Hb - rxn9_k2*IF*H_in)
p_ref[16] = k_H_off_range[0]  # rxn9_k1
p_ref[17] = k_H_on_range[0]   # rxn9_k2   

# rxn10: IF + S_in -> IF_Sb; vol*(rxn10_k1*IF*S_in - rxn10_k2*IF_Sb)
p_ref[18] = k_S_on_range[0]   # rxn10_k1
p_ref[19] = k_S_off_range[0]   # rxn10_k2    

# rxn11: IF_Hb + S_in -> IF_Hb_Sb; vol*(rxn11_k1*IF_Hb*S_in - rxn11_k2*IF_Hb_Sb)
p_ref[20] = k_S_on_range[0] # rxn11_k1
p_ref[21] = k_S_off_range[0]   # rxn11_k2    

# rxn12: IF_Hb_Sb -> IF_Sb + H_in; vol*(rxn12_k1*IF_Hb_Sb - rxn12_k2*IF_Sb*H_in)
p_ref[22] = k_H_off_range[0] # rxn12_k1
p_ref[23] = k_H_on_range[0]  # rxn12_k2    

# experimental noise
p_ref[24] = sigma_range[0] # sigma

p_ref[2] = k_H_on
p_ref[3] = k_H_off
p_ref[4] = k_S_off
p_ref[5] = k_S_on
p_ref[6] = k_conf
p_ref[7] = k_conf
p_ref[10] = k_conf
p_ref[11] = k_conf
p_ref[20] = k_S_on
p_ref[21] = k_S_off
p_ref[22] = k_H_off
p_ref[23] = k_H_on
p_ref[24] = sigma_ref
print(p_ref)




# mcmc trajectory (before burn in)
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
flat_samples = sampler.get_chain(discard=n_burn,  thin=100, flat=True)
print(np.size(flat_samples))
print(np.shape(flat_samples))
logl = sampler.get_log_prob(discard=n_burn, thin=100, flat=True)
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
    plt.plot(y_pred_i, alpha=0.1, color='grey')
plt.title('y_pred and y_obs')
plt.ylabel('y')
plt.xlabel('x')
plt.plot(y_obs, ls='None', color='black', marker='o',label="observed")
plt.legend(fontsize=14)
plt.savefig(new_dir/f'{filename}_example_plots.png')

fig, axes = plt.subplots(5, 5, figsize=(10,10))
ax = axes.flatten()

d_dict = {}

ref_idx_list = [2,3,4,5,6,7,10,11,20,21,22,23,24]
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