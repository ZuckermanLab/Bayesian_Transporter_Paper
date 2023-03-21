import tellurium as te
import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
import psutil
import roadrunner
import ray
from ray.util.multiprocessing import Pool as ray_Pool
from ray.util import ActorPool


def run_synthetic_ssme_assay(rr: roadrunner.RoadRunner):
    '''perform a synthetic solid-supported membrane electrophysiology assay'''
    m = rr
    t_stage = 1
    n_pts_per_stage = 60

    H_out_sequence = [1e-7,5e-7,1e-7]  # fixed buffer solutions for ion, for n stages (e.g. 3 stages)
    S_out_sequence = [1e-3,1e-3,1e-3]  # fixed buffer solutions for substate, for n stages (e.g. 3 stages)
    buffer_solution_sequence = zip(H_out_sequence, S_out_sequence)

    m.resetToOrigin()
    m.integrator.absolute_tolerance = 1e-22
    m.integrator.relative_tolerance = 1e-12
    selections = ['time', 'current']
    results = []

    for i, solution in enumerate(buffer_solution_sequence):
        setattr(m, 'H_out', solution[0])
        setattr(m, 'S_out', solution[1])
        results.append(m.simulate(i,i+t_stage,n_pts_per_stage, selections=selections))
    return np.vstack(results).T


def run_synthetic_ssme_assay_w_events(rr_model):
    '''perform a synthetic solid-supported membrane electrophysiology assay'''
    m = rr_model
    t_stage = 1
    n_pts_per_stage = 60

    H_act = 5e-7  # fixed buffer solutions for ion, for n stages (e.g. 3 stages)
    S_act = 1e-3  # fixed buffer solutions for substate, for n stages (e.g. 3 stages)

    selections = ['time', 'current']
    m.resetToOrigin()
    m.integrator.absolute_tolerance = 1e-22
    m.integrator.relative_tolerance = 1e-12
    results = []

    setattr(m, 'H_act', H_act)
    setattr(m, 'S_act', S_act)
    results = m.simulate(0,3,180, selections=selections)
    return np.vstack(results).T


def plot_results(results, x=0, y=1, label='test'):
    plt.figure(figsize=(12,10))
    plt.plot(results[x], results[y])
    plt.savefig(f'{label}.png')


def add_noise(data, noise_stdev):
    y = data
    y_noise = data + np.random.normal(0, noise_stdev, np.size(y))
    return y_noise


def f_wrapper(rr_string):
    rr_model = te.loada(rr_string)
    return run_synthetic_ssme_assay(rr_model)


if __name__ == '__main__':
    model_file = '/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/scratch/antiporter_12D_model_3c_no_events.txt'
    rr_model = te.loada(model_file)
    results = run_synthetic_ssme_assay(rr_model)
    plot_results(results, 0,1, 'test1')
    

    n_trials = int(1e4)
    t0 = time.time()
    for i in range(n_trials):
        _ = run_synthetic_ssme_assay(rr_model)
    tf = time.time()
    trials_per_sec = n_trials/(tf-t0)

    print(f'it took {tf-t0} sec to run {n_trials} synthetic experiments')
    print(f'{trials_per_sec} synthetic experiments / sec')


    model_file2 = '/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/scratch/antiporter_12D_model_3c_events.txt'
    rr_model2 = te.loada(model_file2)
    results2 = run_synthetic_ssme_assay_w_events(rr_model2)
    plot_results(results2, 0,1, 'test2')
   
    
    t0 = time.time()
    for i in range(n_trials):
        _ = run_synthetic_ssme_assay_w_events(rr_model2)
    tf = time.time()
    trials_per_sec = n_trials/(tf-t0)

    print(f'it took {tf-t0} sec to run {n_trials} synthetic experiments - with events')
    print(f'{trials_per_sec} synthetic experiments / sec - with events')

    print(f'max parallel processes: {psutil.cpu_count(logical=False)}')


    n_processes=6
    t0 = time.time()
    with mp.Pool(processes=n_processes) as pool:
        results_p = pool.map(
            f_wrapper,
            [model_file for i in range(n_trials)]
        )
    tf = time.time()
    trials_per_sec = n_trials/(tf-t0)
    print(f'it took {tf-t0} sec to run {n_trials} synthetic experiments - parallel with {n_processes} processes')
    print(f'{trials_per_sec} synthetic experiments / sec - parallel with {n_processes} processes')

    # n_list = [1,2,3,4,5,6]
    # for n in n_list:
    #     n_processes = n
    #     t0 = time.time()
    #     results = []
    #     with mp.Pool(processes=n_processes) as pool:
    #         results = pool.map(
    #             run_synthetic_ssme_assay,
    #             [rr_model for i in range(n_trials)]
    #         )
    #     tf = time.time()
    #     trials_per_sec = n_trials/(tf-t0)
        
    #     print(f'it took {tf-t0} sec to run {n_trials} synthetic experiments - parallel with {n_processes} processes')
    #     print(f'{trials_per_sec} synthetic experiments / sec - parallel with {n_processes} processes')
    
    plot_results(results_p[0], 0,1,'test_parallel_0')
    plot_results(results_p[-1], 0,1,'test_parallel_f')
