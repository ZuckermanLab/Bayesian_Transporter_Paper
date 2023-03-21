import tellurium as te
import numpy as np
import matplotlib.pyplot as plt
import time
import multiprocessing as mp
import psutil
import roadrunner
import scipy as sp
import ray
from ray.util.multiprocessing import Pool as ray_Pool
from ray.util import ActorPool
import SALib as sa



def run_synthetic_ssme_assay(rr: roadrunner.RoadRunner, n_points: int, k_values:list):
    '''perform a synthetic solid-supported membrane electrophysiology assay'''
    m = rr
    t_stage = 1
    n_pts_per_stage = n_points #60

    H_out_sequence = [1e-7,5e-7,1e-7]  # fixed buffer solutions for ion, for n stages (e.g. 3 stages)
    S_out_sequence = [1e-3,1e-3,1e-3]  # fixed buffer solutions for substate, for n stages (e.g. 3 stages)
    buffer_solution_sequence = zip(H_out_sequence, S_out_sequence)

    m.resetToOrigin()
    m.integrator.absolute_tolerance = 1e-22
    m.integrator.relative_tolerance = 1e-12
    # m.conservedMoietyAnalysis = True
    k_dict = {
        'k1_f' : 10**k_values[0],
        'k1_r' : 10**k_values[1],
        'k2_f' : 10**k_values[2],
        'k2_r' : 10**k_values[3],
        'k3_f' : 10**k_values[4],
        'k3_r' : 10**k_values[5],
        'k4_f' : 10**k_values[6],
        'k4_r' : 10**k_values[7],
        'k5_f' : 10**k_values[8],
        'k5_r' : 10**k_values[9],
        'k6_f' : 10**k_values[10],
    }

    for k in k_dict:
        setattr(m, k, k_dict[k])


    test_labels = ['OF', 'OF_Hb', 'IF_Hb', 'IF_Hb_Sb', 'IF_Sb', 'OF_Sb',
                   'H_out', 'H_in', 'S_out', 'S_in'
    ]

    selections = ['time', 'current', 'OF', 'OF_Hb', 'IF_Hb', 'IF_Hb_Sb', 'IF_Sb', 'OF_Sb',
                   'H_out', 'H_in', 'S_out', 'S_in']
    results = []

    for i, solution in enumerate(buffer_solution_sequence):
        
        # set buffer solution
        setattr(m, 'H_out', solution[0])
        setattr(m, 'S_out', solution[1])

        # initialize other concentrations --> check here, something weird w/ concentrations and ouput
        if i==0:
            setattr(m, 'OF', 0.0011694210430300167)
            setattr(m, 'OF_Hb', 0.0)
            setattr(m, 'IF_Hb', 0.0)
            setattr(m, 'IF_Hb_Sb', 0.0)
            setattr(m, 'IF_Sb', 0.0)
            setattr(m, 'OF_Sb', 0.0)
            setattr(m, 'H_in', 1e-7)
            setattr(m, 'S_in', 1e-3)
            # for label in test_labels:
            #     print(getattr(m, label))
            m.simulate(i,i+t_stage,n_pts_per_stage, selections=selections)
            # rr.steadyStateSelectionss =ss_labels
            # ss_values = rr.getSteadyStateValues()
            # for j, ss_label in enumerate(ss_labels):
            #     setattr(m, ss_label, ss_values[j])
        else:
            results.append(m.simulate(i,i+t_stage,n_pts_per_stage, selections=selections))

        # for label in test_labels:
        #     print(getattr(m, label))
        # print('----------')
        

        # results.append(m.simulate(i,i+t_stage,n_pts_per_stage, selections=selections))
        # if i ==0:
        #     print(results)
        #results.append(m.simulate(0,1,n_pts_per_stage, selections=selections))
    #print('\n')
    return np.vstack(results).T


@ray.remote
class SimulatorActor(object):
    """Ray actor to execute simulations."""

    def __init__(self, model_string, y_obs, n_points):
        self.rr = te.loada(model_string)
        self.y_obs = y_obs
        self.n_points = n_points


    def simulate(self):
        '''perform a synthetic solid-supported membrane electrophysiology assay'''
        m = self.rr
        t_stage = 1
        n_pts_per_stage = self.n_points #60

        H_out_sequence = [1e-7,5e-7,1e-7]  # fixed buffer solutions for ion, for n stages (e.g. 3 stages)
        S_out_sequence = [1e-3,1e-3,1e-3]  # fixed buffer solutions for substate, for n stages (e.g. 3 stages)
        buffer_solution_sequence = zip(H_out_sequence, S_out_sequence)

        m.resetToOrigin()
        m.integrator.absolute_tolerance = 1e-22
        m.integrator.relative_tolerance = 1e-12
        # m.conservedMoietyAnalysis = True

        test_labels = ['OF', 'OF_Hb', 'IF_Hb', 'IF_Hb_Sb', 'IF_Sb', 'OF_Sb',
                    'H_out', 'H_in', 'S_out', 'S_in'
        ]

        selections = ['time', 'current', 'OF', 'OF_Hb', 'IF_Hb', 'IF_Hb_Sb', 'IF_Sb', 'OF_Sb',
                    'H_out', 'H_in', 'S_out', 'S_in']
        results = []

        for i, solution in enumerate(buffer_solution_sequence):
            
            # set buffer solution
            setattr(m, 'H_out', solution[0])
            setattr(m, 'S_out', solution[1])

            # initialize other concentrations --> check here, something weird w/ concentrations and ouput
            if i==0:
                setattr(m, 'OF', 0.0011694210430300167)
                setattr(m, 'OF_Hb', 0.0)
                setattr(m, 'IF_Hb', 0.0)
                setattr(m, 'IF_Hb_Sb', 0.0)
                setattr(m, 'IF_Sb', 0.0)
                setattr(m, 'OF_Sb', 0.0)
                setattr(m, 'H_in', 1e-7)
                setattr(m, 'S_in', 1e-3)
                # for label in test_labels:
                #     print(getattr(m, label))
                m.simulate(i,i+t_stage,n_pts_per_stage, selections=selections)
                # rr.steadyStateSelectionss =ss_labels
                # ss_values = rr.getSteadyStateValues()
                # for j, ss_label in enumerate(ss_labels):
                #     setattr(m, ss_label, ss_values[j])
            else:
                results.append(m.simulate(i,i+t_stage,n_pts_per_stage, selections=selections))

            # for label in test_labels:
            #     print(getattr(m, label))
            # print('----------')
            

            # results.append(m.simulate(i,i+t_stage,n_pts_per_stage, selections=selections))
            # if i ==0:
            #     print(results)
            #results.append(m.simulate(0,1,n_pts_per_stage, selections=selections))
        #print('\n')
        return np.vstack(results).T
    
    def get_logl(self, x=0):
        D = self.simulate()
        y_obs = self.y_obs
        y_pred = D[1]
        sigma = 1e-10
        logl = np.sum(np.log(sp.stats.norm.pdf(y_obs, y_pred, sigma)))
        return logl


def plot_results(results, x=0, y=1, label='test'):
    plt.figure(figsize=(12,10))
    plt.plot(results[y][:], 'o', alpha=0.75)
    plt.savefig(f'{label}.png')


def transporter_ODE(t, y, H_out, S_out):

    v_out = 1.048e-07
    v_m = 1.42e-08
    v_in = 5.24e-08

    k1_f = 1e10
    k1_r = 1e3
    k2_f = 1e2
    k2_r = 1e2
    k3_f = 1e7
    k3_r = 1e3
    k4_f = 1e3
    k4_r = 1e10
    k5_f = 1e2
    k5_r = 1e2
    k6_f = 1e3;
    k6_r = (k1_f*k2_f*k3_f*k4_f*k5_f*k6_f)/(k1_r*k2_r*k3_r*k4_r*k5_r)

    OF = y[0]
    OF_Hb = y[1]
    IF_Hb = y[2]
    IF_Hb_Sb = y[3]
    IF_Sb = y[4]
    OF_Sb = y[5]
    H_in = y[6]
    S_in = y[7]

    vrxn1 = v_out*(k1_f*OF*H_out-k1_r*OF_Hb)
    vrxn2 = v_m*(k2_f*OF_Hb-k2_r*IF_Hb)
    vrxn3 = v_in*(k3_f*IF_Hb*S_in-k3_r*IF_Hb_Sb)
    vrxn4 = v_in*(k4_f*IF_Hb_Sb-k4_r*IF_Sb*H_in)
    vrxn5 = v_m*(k5_f*IF_Sb-k5_r*OF_Sb)
    vrxn6 = v_out*(k6_f*OF_Sb-k6_r*OF*S_out)

    dOF_dt = (-vrxn1 + vrxn6)/v_m
    dOF_Hb_dt = (vrxn1 - vrxn2)/v_m
    dIF_Hb_dt = (vrxn2 - vrxn3)/v_m
    dIF_Hb_Sb_dt = (vrxn3 - vrxn4)/v_m
    dIF_Sb_dt = (vrxn4 - vrxn5)/v_m
    dOF_Sb_dt = (vrxn5 - vrxn6)/v_m
    dH_in_dt = (vrxn4)/v_in
    dS_in_dt = (-vrxn3)/v_in

    y_t = [dOF_dt, dOF_Hb_dt, dIF_Hb_dt, dIF_Hb_Sb_dt, dIF_Sb_dt, dOF_Sb_dt, dH_in_dt, dS_in_dt]

    return y_t


def ray_example2(model_string, y_obs, n_points, N_trials, N_actors):
    actor_count = N_actors 
    simulators = [SimulatorActor.remote(model_string, y_obs, n_points) for _ in range(actor_count)]

    f = lambda a,b : a.get_logl.remote(b)
    pool = ActorPool(simulators)
    gen = pool.map(f, [i for i in range(N_trials)])
    return list(gen)


def calc_norm_log_like(mu,sigma,X):
    ''' calculates the Normal log-likelihood function: -[(n/2)ln(2pi*sigma^2)]-[sum((X-mu)^2)/(2*sigma^2)]
    ref: https://www.statlect.com/fundamentals-of-statistics/normal-distribution-maximum-likelihood 
    '''
    # fix this - remove loop
    n = len(X)
    f1 = -1*(n/2)*np.log(2*np.pi*sigma**2)
    f2_a = -1/(2*sigma**2)
    f2_b = 0 
    for i in range(n):
        f2_b += (X[i]-mu[i])**2
    f2 = f2_a*f2_b
    log_likelihood = f1+f2
    return log_likelihood


def calc_norm_log_like_v2(mu,sigma,X):
    ''' calculates the Normal log-likelihood function: -[(n/2)ln(2pi*sigma^2)]-[sum((X-mu)^2)/(2*sigma^2)]
    ref: https://www.statlect.com/fundamentals-of-statistics/normal-distribution-maximum-likelihood 
    '''
    #logl = np.sum(np.log(sp.stats.norm.pdf(X, mu, sigma)))
    logl = sp.stats.norm.logpdf(X, mu, sigma).sum()
    return logl


def get_logl(rr, n_points, y_obs, k_list):
        rate_constants = k_list[:-1]
        sigma = 10**k_list[-1]
        try:
            D = run_synthetic_ssme_assay(rr, n_points, rate_constants)
        except:
            return -1e100
        y_pred = D[1]
        #logl = calc_norm_log_like(y_pred, sigma, y_obs)
        logl = calc_norm_log_like_v2(y_pred, sigma, y_obs)
        return logl


def scipy_ODE_solver(n_points):
    # ode 
    y_0 = [0.0011694210430300167,
           0,
           0,
           0,
           0,
           0,
           1e-7, 
           1e-3]
    t_span = [0,1]
    t_eval = np.linspace(0,1,n_points)
    v_out = 1.048e-07
    v_m = 1.42e-08
    v_in = 5.24e-08

    k1_f = 1e10
    k1_r = 1e3
    k2_f = 1e2
    k2_r = 1e2
    k3_f = 1e7
    k3_r = 1e3
    k4_f = 1e3
    k4_r = 1e10
    k5_f = 1e2
    k5_r = 1e2
    k6_f = 1e3;
    k6_r = (k1_f*k2_f*k3_f*k4_f*k5_f*k6_f)/(k1_r*k2_r*k3_r*k4_r*k5_r)
    
    elem_charge = 1.602e-19
    N_av = 6.022e23
    
    sol1 = sp.integrate.solve_ivp(transporter_ODE, t_span, y_0, args=(1e-7,1e-3), method='LSODA', dense_output=False, t_eval=t_eval, rtol=1e-8, atol=1e-12)
    sol2 = sp.integrate.solve_ivp(transporter_ODE, t_span, np.transpose(sol1.y)[-1], args=(5e-7,1e-3), method='LSODA', dense_output=False, t_eval=t_eval, rtol=1e-8, atol=1e-12)
    sol3 = sp.integrate.solve_ivp(transporter_ODE, t_span, np.transpose(sol2.y)[-1], args=(1e-7,1e-3), method='LSODA', dense_output=False, t_eval=t_eval, rtol=1e-8, atol=1e-12)
    y_sol = np.hstack([sol1.y, sol2.y, sol3.y])
    OF = y_sol[0]
    OF_Hb = y_sol[1]
    IF_Hb = y_sol[2]
    IF_Hb_Sb = y_sol[3]
    IF_Sb = y_sol[4]
    OF_Sb = y_sol[5]
    H_in = y_sol[6]
    S_in = y_sol[7]
    H_in_net_flux = v_in*(k4_f*IF_Hb_Sb-k4_r*IF_Sb*H_in)
    current = H_in_net_flux*elem_charge*N_av
    return current
    
    


if __name__ == '__main__':

    # modelfile = '/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/scratch/antiporter_12D_model_3c_no_events.txt'
    sbmlfile = '/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/scratch/antiporter_12D_model_3c_no_events.xml'
    rr = roadrunner.RoadRunner(sbmlfile)
    # # print(te.getODEsFromModel(rr))

    k_ref = [10,3,2,2,7,3,3,10,2,2,3]
    p_ref = [10,3,2,2,7,3,3,10,2,2,3, -10]
    print(p_ref)

    results = run_synthetic_ssme_assay(rr, 500, k_ref)
    y_true = results[1]
    y_obs = results[1] + np.random.normal(0, 1e-10, np.size(results[1])) 
   
    plt.figure(figsize=(12,10))
    plt.plot(y_obs, 'o', alpha=0.75)
    plt.savefig(f'test1.png')

    # plot_results(results, 0,1, 'test1')
    print(get_logl(rr, 500, y_obs, p_ref))

 

    n_sobol_points = int(2**16)
    p = sa.ProblemSpec({
        'names': ['k1_f', 'k1_r','k2_f', 'k2_r','k3_f', 'k3_r','k4_f', 'k4_r','k5_f', 'k5_r','k6_f', 'sigma'],
        'bounds':
            [ [6,12], [-1, 5], [-2, 4], [-2, 4], [3, 9], [-1, 5],
              [-1, 5], [6, 12], [-2, 4], [-2, 4], [-1, 5], [-11, -9],         
            ],
    })

    def wrapped_ssme(rr, n_points, K, y_true):
        y_pred= run_synthetic_ssme_assay(rr, n_points, K)[1]
        rmse = np.sqrt(np.mean(np.square(y_true-y_pred)))
        return rmse
    print(wrapped_ssme(rr, 500, p_ref, y_true))
    p.sample_sobol(n_sobol_points, calc_second_order=True)
    param_values = p.samples
    np.savetxt("param_values.txt", param_values)
    t0 = time.time()

    Y = np.array([wrapped_ssme(rr, 500, X, y_true) for X in param_values])
    p.set_results(Y)
    p.analyze_sobol()
    print(p)

    axes = p.plot()
    axes[0].set_yscale('log')
    fig = plt.gcf()  # get current figure
    fig.set_size_inches(25, 5)
    plt.tight_layout()
    plt.savefig('sensitivity_v2.png')
    assert(1==0)
   

    plt.figure(figsize=(12,10))
    plt.title('ODE integration performance - libroadrunner vs scipy')
    plt.ylabel('wall clock (s)')
    plt.xlabel('N (number of data points)')

    n_trials = int(1e4)
    wall_clock_time = []
    wall_clock_stdev = []
    n_points_list = [10,50,100,500,1000]
    for i in n_points_list:
        n_points = i
        D = run_synthetic_ssme_assay(rr, n_points)
        y_obs = D[1]
        t0 = time.time()
        for j in range(n_trials): 
             _ = run_synthetic_ssme_assay(rr, n_points)
        tf = time.time()
        wall_clock_time.append(tf-t0)
    plt.plot([3*i for i in n_points_list], wall_clock_time)
    plt.plot([3*i for i in n_points_list], wall_clock_time, 'o', label='1e4 iterations')

    n_trials = int(1e3)
    wall_clock_time = []
    wall_clock_stdev = []
    n_points_list = [10,50,100,500,1000]
    for i in n_points_list:
        n_points = i
        t0 = time.time()
        for j in range(n_trials): 
            _ = run_synthetic_ssme_assay(rr, n_points)
        tf = time.time()
        wall_clock_time.append(tf-t0)

    plt.plot([3*i for i in n_points_list], wall_clock_time)
    plt.plot([3*i for i in n_points_list], wall_clock_time, 'o', label='1e3 iterations - libroadrunner (CVODE)')

    n_trials = int(1e2)
    wall_clock_time = []
    wall_clock_stdev = []
    n_points_list = [10,50,100,500,1000]
    for i in n_points_list:
        n_points = i
        t0 = time.time()
        for j in range(n_trials): 
            _ = run_synthetic_ssme_assay(rr, n_points)
        tf = time.time()
        wall_clock_time.append(tf-t0)

    plt.plot([3*i for i in n_points_list], wall_clock_time)
    plt.plot([3*i for i in n_points_list], wall_clock_time, 'o', label='1e2 iterations - libroadrunner (CVODE)')
    plt.legend()
    #plt.ylim(0,60)

    n_trials = int(1e4)
    wall_clock_time = []
    wall_clock_stdev = []
    n_points_list = [10,50,100,500,1000]
    for i in n_points_list:
        n_points = i
        t0 = time.time()
        for j in range(n_trials): 
            _ = scipy_ODE_solver(n_points)
        tf = time.time()
        wall_clock_time.append(tf-t0)

    plt.plot([3*i for i in n_points_list], wall_clock_time)
    plt.plot([3*i for i in n_points_list], wall_clock_time, 'o', label='1e4 iterations - Scipy ODE (LSODA)')


    n_trials = int(1e3)
    wall_clock_time = []
    wall_clock_stdev = []
    n_points_list = [10,50,100,500,1000]
    for i in n_points_list:
        n_points = i
        t0 = time.time()
        for j in range(n_trials): 
            _ = scipy_ODE_solver(n_points)
        tf = time.time()
        wall_clock_time.append(tf-t0)

    plt.plot([3*i for i in n_points_list], wall_clock_time)
    plt.plot([3*i for i in n_points_list], wall_clock_time, 'o', label='1e3 iterations - Scipy ODE (LSODA)')

    n_trials = int(1e2)
    wall_clock_time = []
    wall_clock_stdev = []
    n_points_list = [10,50,100,500,1000]
    for i in n_points_list:
        n_points = i
        t0 = time.time()
        for j in range(n_trials): 
            _ = scipy_ODE_solver(n_points)
        tf = time.time()
        wall_clock_time.append(tf-t0)

    plt.plot([3*i for i in n_points_list], wall_clock_time)
    plt.plot([3*i for i in n_points_list], wall_clock_time, 'o', label='1e2 iterations - Scipy ODE (LSODA)')
    plt.legend()
    #plt.ylim(0,100)
    plt.savefig('test_ODE_runs.png')    
    assert(1==0)



    # r = ray_example(modelfile)
    # #print(r)
    # plot_results(r[0], 0,1, 'test_ray1')
    # plot_results(r[1], 0,1, 'test_ray2')

    n_trials = int(1e4)
    times = []
    n_points_list = [10,50,100,500,1000]
    plt.figure(figsize=(12,10))
    plt.title('Log-likelihood performance - updated (Ray)')
    plt.ylabel('wall clock (s)')
    plt.xlabel('N (number of data points)')

    for i in range(1,7):
        n_processes = i
        times = []
        for j in n_points_list:
            D = run_synthetic_ssme_assay(rr, n_points)
            y_obs = D[1]
            ray.init(log_to_driver=False, num_cpus=i)
            t0 = time.time()
            r2 = ray_example2(modelfile, y_obs, n_points, n_trials, i)
            tf = time.time()
            ray.shutdown()
            times.append(tf-t0)
        plt.plot([3*k for k in n_points_list], times)
        plt.plot([3*k for k in n_points_list], times, 'o', label=f'{i} parallel processes')
    
    plt.legend()
    plt.ylim(0,60)
    plt.savefig('test_logl_performance_v2_parallel.png')    
        
          
    
