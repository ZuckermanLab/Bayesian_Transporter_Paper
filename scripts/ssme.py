import utility
import roadrunner 
import numpy as np


def simulate_y_pred_rr(rr:roadrunner.RoadRunner, k:list, **kwargs):  
    '''perform a synthetic solid-supported membrane electrophysiology assay'''
    default_kwargs = {'n_points_stage': 500,
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
                      'cutoff_stage': 250, 
                      }
    
    kwargs = { **default_kwargs, **kwargs }

    # get ssme parameters
    m = rr 
    exp_K = np.power(10, k)
    k_dict = dict(zip(kwargs['k_labels'], exp_K))  # K is in log10 scale
    events = kwargs['events']
    t_stage = kwargs['t_stage']
    n_pts_per_stage = kwargs['n_points_stage']
    buffer_labels = kwargs['buffer_labels']  # names of buffer solutions (e.g. H_out and S_out)
    buffer_sequence = kwargs['buffer_sequence'] # fixed buffer solutions for ion, for n stages (e.g. 3 stages)
    n_stages = len(buffer_sequence)
    t_end = t_stage*n_stages
    n_pts_total = n_pts_per_stage*n_stages
    states = kwargs['state_labels']
    state_init = kwargs['state_init']
    a_tol = kwargs['a_tol']
    r_tol = kwargs['r_tol']
    selections = kwargs['output_label']
    cutoff = kwargs['cutoff_stage']

    # reset model and set tolerances
    m.resetToOrigin()
    m.integrator.absolute_tolerance = a_tol
    m.integrator.relative_tolerance = r_tol

    # update model rate constants
    for k in k_dict:
        setattr(m, k, float(k_dict[k]))

    # start synthetic SSME experiment
    results = [] 
    if events == False:  # don't use built-in SBML events
        for i, solution in enumerate(buffer_sequence):  # update buffer solution for each assay stage 
            for j, label in enumerate(buffer_labels):
                setattr(m, label, solution[j])  
            if i==0:  # set initial state concentrations for stage 1 equilibration
                for j, label in enumerate(states):
                    setattr(m, label, state_init[j])
                m.simulate(i,i+t_stage,n_pts_per_stage, selections=selections)
            else:
                tmp = m.simulate(i,i+t_stage,n_pts_per_stage, selections=selections)
                results.append(tmp[2:cutoff])
        return np.vstack(results).T    
    else:
        results = [m.simulate(0,t_end,n_pts_total, selections=selections)]
        return np.vstack(results).T  
    

if __name__ == '__main__':
    fname = '/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/scratch/antiporter_12D_model_3c_no_events.xml'
    r = utility.load_rr_model_from_sbml(fname)
    k = [10,3,2,2,7,3,3,10,2,2,3]
    res = simulate_y_pred_rr(r, k, n_points_stage=200)
    t = res[0]
    y = res[1]

    import matplotlib.pyplot as plt
    plt.plot(t,y)
    plt.savefig('test_new.png')