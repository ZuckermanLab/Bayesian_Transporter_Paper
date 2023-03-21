import roadrunner
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import SALib as sa
import time
import scipy as sp


def simulate_roadrunner_ODE(rr:roadrunner.RoadRunner, K:list, n_points:int, events:bool):  
    '''perform a synthetic solid-supported membrane electrophysiology assay'''
    m = rr
    t_stage = 1
    n_pts_per_stage = n_points #60

    H_out_sequence = [1e-7,5e-7,1e-7]  # fixed buffer solutions for ion, for n stages (e.g. 3 stages)
    S_out_sequence = [1e-3,1e-3,1e-3]  # fixed buffer solutions for substate, for n stages (e.g. 3 stages)
    buffer_solution_sequence = list(zip(H_out_sequence, S_out_sequence))
    n_stages = len(buffer_solution_sequence)
    t_end = t_stage*n_stages
    n_pts_total = n_pts_per_stage*n_stages

    m.resetToOrigin()
    m.integrator.absolute_tolerance = 1e-22
    m.integrator.relative_tolerance = 1e-12
    # m.conservedMoietyAnalysis = True
    k_dict = {
        'k1_f' : 10**K[0],
        'k1_r' : 10**K[1],
        'k2_f' : 10**K[2],
        'k2_r' : 10**K[3],
        'k3_f' : 10**K[4],
        'k3_r' : 10**K[5],
        'k4_f' : 10**K[6],
        'k4_r' : 10**K[7],
        'k5_f' : 10**K[8],
        'k5_r' : 10**K[9],
        'k6_f' : 10**K[10],
    }

    for k in k_dict:
        setattr(m, k, k_dict[k])

    selections = ['time', 'current', 'OF', 'OF_Hb', 'IF_Hb', 'IF_Hb_Sb', 'IF_Sb', 'OF_Sb',
                   'H_out', 'H_in', 'S_out', 'S_in']
    results = []

    if events == False:
        for i, solution in enumerate(buffer_solution_sequence):
            print(solution)
            # set buffer solution
            setattr(m, '[H_out]', solution[0])
            setattr(m, '[S_out]', solution[1])

            # initialize other concentrations --> check here, something weird w/ concentrations and ouput
            if i==0:
                setattr(m, '[OF]', 0.0011694210430300167)
                setattr(m, '[OF_Hb]', 0.0)
                setattr(m, '[IF_Hb]', 0.0)
                setattr(m, '[IF_Hb_Sb]', 0.0)
                setattr(m, '[IF_Sb]', 0.0)
                setattr(m, '[OF_Sb]', 0.0)
                setattr(m, '[H_in]', 1e-7)
                setattr(m, '[S_in]', 1e-3)
                results.append(m.simulate(i,i+t_stage,n_pts_per_stage, selections=selections))
            else:
                results.append(m.simulate(i,i+t_stage,n_pts_per_stage, selections=selections))
        return np.vstack(results).T    
    else:
        results = [m.simulate(0,t_end,n_pts_total, selections=selections)]
        return np.vstack(results).T  
    

def simulate_roadrunner_ODE_ss(rr:roadrunner.RoadRunner, K:list, n_points:int):  
    '''perform a synthetic solid-supported membrane electrophysiology assay'''
    m = rr
    t_stage = 1
    n_pts_per_stage = n_points #60

    H_out_sequence = [1e-7,5e-7,1e-7]  # fixed buffer solutions for ion, for n stages (e.g. 3 stages)
    S_out_sequence = [1e-3,1e-3,1e-3]  # fixed buffer solutions for substate, for n stages (e.g. 3 stages)
    buffer_solution_sequence = list(zip(H_out_sequence, S_out_sequence))

    m.resetToOrigin()
    m.integrator.absolute_tolerance = 1e-22
    m.integrator.relative_tolerance = 1e-12
    k_dict = {
        'k1_f' : 10**K[0],
        'k1_r' : 10**K[1],
        'k2_f' : 10**K[2],
        'k2_r' : 10**K[3],
        'k3_f' : 10**K[4],
        'k3_r' : 10**K[5],
        'k4_f' : 10**K[6],
        'k4_r' : 10**K[7],
        'k5_f' : 10**K[8],
        'k5_r' : 10**K[9],
        'k6_f' : 10**K[10],
    }

    for k in k_dict:
        setattr(m, k, k_dict[k])

    selections = ['time', 'current', 'OF', 'OF_Hb', 'IF_Hb', 'IF_Hb_Sb', 'IF_Sb', 'OF_Sb',
                   'H_out', 'H_in', 'S_out', 'S_in']
    test_selections = ['current', 'rxn4', 'OF', 'OF_Hb', 'IF_Hb', 'IF_Hb_Sb', 'IF_Sb', 'OF_Sb',
                   'H_out', 'H_in', 'S_out', 'S_in']
    results = []
    #ss_selections = ['[OF]', '[OF_Hb]', '[IF_Hb]', '[IF_Hb_Sb]', '[IF_Sb]', '[OF_Sb]',
    #               '[H_in]', '[S_in]']
    ss_selections = m.getSteadyStateSelectionStrings()
    m.steadyStateSelections = ss_selections
    print(m.steadyStateSelections)

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
            # results.append(m.simulate(i,i+t_stage,n_pts_per_stage, selections=selections))

            
            
            for s in test_selections:    
                print(f'before ss: {s}={getattr(m,s)}')
            print(f'before ss: v_in*(k4_f*IF_Hb_Sb - k4_r*IF_Sb*H_in) = \n{m.v_in}*({m.k4_f}*{m.IF_Hb_Sb} - \n{m.k4_r}*{m.IF_Sb}*{m.H_in})\n')
            
            ## steady state solver
    
            m.conservedMoietyAnalysis = True
            m.getSteadyStateSolver().relative_tolerance = 1e-16
            ss_values = m.getSteadyStateValuesNamedArray()
         
            
            # setattr(m, 'OF', ss_values['[OF]'][0])
            # setattr(m, 'OF_Hb', ss_values['[OF_Hb]'][0])
            # setattr(m, 'IF_Hb', ss_values['[IF_Hb]'][0])
            # setattr(m, 'IF_Hb_Sb', ss_values['[IF_Hb_Sb]'][0])
            # setattr(m, 'IF_Sb', ss_values['[IF_Sb]'][0])
            # setattr(m, 'OF_Sb', ss_values['[OF_Sb]'][0])
            # setattr(m, 'H_in', ss_values['[H_in]'][0])
            # setattr(m, 'S_in', ss_values['[S_in]'][0])

            # or solve ODE directly (debugging)
            m.simulate(i,i+t_stage,n_pts_per_stage, selections=selections) 

            for s in test_selections:
                print(f'after ss: {s}={getattr(m,s)}')
            print(f'after ss: v_in*(k4_f*IF_Hb_Sb - k4_r*IF_Sb*H_in) = \n{m.v_in}*({m.k4_f}*{m.IF_Hb_Sb} - \n{m.k4_r}*{m.IF_Sb}*{m.H_in})\n')
            print('\n')
        else:
            for s in test_selections:
                print(f'stage {i+1} before: {s}={getattr(m,s)}')
            print(f'stage {i+1} before: v_in*(k4_f*IF_Hb_Sb - k4_r*IF_Sb*H_in) = {m.v_in}*({m.k4_f}*{m.IF_Hb_Sb} - {m.k4_r}*{m.IF_Sb}*{m.H_in})')
            results.append(m.simulate(i,i+t_stage,n_pts_per_stage, selections=selections))
            for s in test_selections:
                print(f'stage {i+1} after: {s}={getattr(m,s)}')
            print(f'stage {i+1} after: v_in*(k4_f*IF_Hb_Sb - k4_r*IF_Sb*H_in) = {m.v_in}*({m.k4_f}*{m.IF_Hb_Sb} - {m.k4_r}*{m.IF_Sb}*{m.H_in})')
            print('\n')
            
    return np.vstack(results).T    
 
    

def calc_logl(rr:roadrunner.RoadRunner(), K:list, n_points:int, events:bool, y_obs:list,  sigma:float, spy:bool):
    try:
        results = simulate_roadrunner_ODE(rr, K, n_points, events)
    except:
        return -1e30
    y = results[1]
    mu = y
    sigma = 10**sigma
    X = y_obs

    if spy == False:
        n = len(X)
        f1 = -1*(n/2)*np.log(2*np.pi*sigma**2)
        f2_a = -1/(2*sigma**2)
        f2_b = 0 
        for i in range(n):
            f2_b += (X[i]-mu[i])**2
        f2 = f2_a*f2_b
        log_likelihood = f1+f2
    else:
        log_likelihood = sp.stats.norm.logpdf(X, mu, sigma).sum()
    return log_likelihood


def nlogl_wrapper(P_values, rr_model, y_obs):
    n_points = 500
    events = False
    spy = True
    K_list = P_values[:-1]
    sigma = P_values[-1]
    return -1*calc_logl(rr_model, K_list, n_points, events, y_obs, sigma, spy)
   


def simulate_scipy_ODE(n_points:int): 


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
    


def test_events(rr_events, rr_no_events, N_trials_list, N_integration_points_list):

    plt.figure(figsize=(12,10))
    plt.ylabel('wall clock (s)')
    plt.xlabel('N ODE integration points (per stage, 3 stages total)')

    for z in range(2):
        for N_trials in N_trials_list:
            y = []
            x = []
            if z == 0:
                label = f'SBML events: {N_trials} trials'
            else:
                label = f'no SBML events: {N_trials} trials'
            K_ref = [10,3,2,2,7,3,3,10,2,2,3]
            K_test = [K_ref]*int(N_trials)
            for N_integration_points in N_integration_points_list:
                x.append(N_integration_points)
                if z == 0:
                    t0 = time.time()
                    for k in range(int(N_trials)):
                        _ = simulate_roadrunner_ODE(rr_events, K_test[k], N_integration_points, events=True)
                    y.append(time.time()-t0)
                else:
                    t0 = time.time()
                    for k in range(int(N_trials)):
                        _ = simulate_roadrunner_ODE(rr_no_events, K_test[k], N_integration_points, events=False)
                    y.append(time.time()-t0)
            plt.plot(x,y, linewidth=2, label=label)
            plt.plot(x,y,'o', color=plt.gca().lines[-1].get_color())

    plt.legend()
    plt.tight_layout()
    plt.savefig('test_events.png')


def test_logl(rr_no_events, N_trials_list, N_integration_points_list):
    
    plt.figure(figsize=(12,10))
    plt.ylabel('wall clock (s)')
    plt.xlabel('N ODE integration points (per stage, 3 stages total)')

    for z in range(2):
        for N_trials in N_trials_list:
            y = []
            x = []
            if z == 0:
                label = f'scipy logpdf: {N_trials} trials'
            else:
                label = f'manual logpdf: {N_trials} trials'
            K_ref = [10,3,2,2,7,3,3,10,2,2,3]
            sigma_ref = -10
            sigma_test = [sigma_ref]*int(N_trials)
            K_test = [K_ref]*int(N_trials)

            for N_integration_points in N_integration_points_list:
                x.append(N_integration_points)
                results = simulate_roadrunner_ODE(rr_no_events, K_ref, N_integration_points, events=False)
                y_obs = results[1] + np.random.normal(0,10**sigma_ref, np.size(results[1]))
                if z == 0:
                    t0 = time.time()
                    for k in range(int(N_trials)):
                        _ = calc_logl(rr_no_events, K_test[k], N_integration_points, events=False, y_obs=y_obs, sigma=sigma_test[k], spy=True)
                    y.append(time.time()-t0)
                else:
                    t0 = time.time()
                    for k in range(int(N_trials)):
                        _ = calc_logl(rr_no_events, K_test[k], N_integration_points, events=False, y_obs=y_obs, sigma=sigma_test[k], spy=False)
                    y.append(time.time()-t0)
            plt.plot(x,y, linewidth=2, label=label)
            plt.plot(x,y,'o', color=plt.gca().lines[-1].get_color())

    plt.legend()
    plt.tight_layout()
    plt.savefig('test_logl.png')



def test_ODE_integration(rr_no_events, N_trials_list, N_integration_points_list):
    
    plt.figure(figsize=(12,10))
    plt.ylabel('wall clock (s)')
    plt.xlabel('N ODE integration points (per stage, 3 stages total)')

    for z in range(2):
        for N_trials in N_trials_list:
            y = []
            x = []
            if z == 0:
                label = f'scipy ODE (LSODA): {N_trials} trials'
            else:
                label = f'libroadrunner (CVODES): {N_trials} trials'
            K_ref = [10,3,2,2,7,3,3,10,2,2,3]
            K_test = [K_ref]*int(N_trials)

            for N_integration_points in N_integration_points_list:
                x.append(N_integration_points)
                if z == 0:
                    t0 = time.time()
                    for k in range(int(N_trials)):
                        _ = simulate_scipy_ODE(N_integration_points)
                        
                    y.append(time.time()-t0)
                else:
                    t0 = time.time()
                    for k in range(int(N_trials)):
                        _ = simulate_roadrunner_ODE(rr_no_events, K_test[k], N_integration_points, events=False)
                    y.append(time.time()-t0)
            plt.plot(x,y, linewidth=2, label=label)
            plt.plot(x,y,'o', color=plt.gca().lines[-1].get_color())

    plt.legend()
    plt.tight_layout()
    plt.savefig('test_ODE_integration.png')
 


if __name__ == '__main__':
    
    sbmlfile_events = '/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/scratch/antiporter_12D_model_3c_events.xml'
    sbmlfile_no_events = '/Users/georgeau/Desktop/GitHub/Bayesian_Transporter/scratch/antiporter_12D_model_3c_no_events.xml'
    rr_events = roadrunner.RoadRunner(sbmlfile_events)
    rr_no_events = roadrunner.RoadRunner(sbmlfile_no_events)

    np.random.seed(0)
    K_ref = [10,3,2,2,7,3,3,10,2,2,3]
    sigma_ref = -10  #log10 sigma
    P_ref = K_ref + [sigma_ref]
    print(P_ref)
    results = simulate_roadrunner_ODE(rr_no_events, K_ref, 500, False)
    y_true = results[1]
    y_ref = results[1] + np.random.normal(0, 10**sigma_ref, np.size(y_true))


    print(calc_logl(rr_no_events, K_ref, 500, False, y_ref, sigma_ref, spy=False))
    print(calc_logl(rr_no_events, K_ref, 500, False, y_ref, sigma_ref, spy=True))

    # testing
    y = y_true
    mu = y
    sigma = 10**sigma_ref
    X = y_ref
    logl = sp.stats.norm.logpdf(X, mu, sigma).sum()
    print(logl)

    p_test = [  9.78029506,   3.35117121,   2.36395624,   1.20418677,   6.65981779,
   3.87041271,   2.08912872,   8.51966452,   1.87063971,   2.13440094,
   3.26617998, -10.00669722]
    
    p_test = [11.78176845,   4.84093166,   0.99630239,   0.99146296,   2.97272319,
   0.04012018,   2.28280014,   9.45184095,   2.1751925,   -3.72930717,
  -2.57220945, -10.01036405]
    

    print(calc_logl(rr_no_events, p_test[:-1], 500, False, y_ref, p_test[-1], spy=False))
    print(calc_logl(rr_no_events, p_test[:-1], 500, False, y_ref, p_test[-1], spy=True))

    y_test = simulate_roadrunner_ODE(rr_no_events, p_test[:-1], 500, False)[1]
    y_test_obs = np.random.normal(0, 10**sigma_ref, np.size(y_test))

    # testing
    y = y_test
    mu = y
    sigma = 10**p_test[-1]
    X = y_ref
    logl = sp.stats.norm.logpdf(X, mu, sigma).sum()
    print(logl)



    plt.figure(figsize=(12,10))
    plt.title('current trace - simplex MLE outlier')
    plt.plot(y_true, label='true + noise')
    plt.plot(y_test, label='pred + noise')
    plt.savefig('mle_simplex_outlier.png')
    assert(1==0)    

    #test_events(rr_events, rr_no_events, [1e2, 1e3, 1e4], [10,50,100,500,1000])
    #test_logl(rr_no_events, [1e2, 1e3, 1e4], [10,50,100,500,1000])
    #test_ODE_integration(rr_no_events, [1e2, 1e3, 1e4], [10,50,100,500,1000])

    # # ss tests
    # results_test = simulate_roadrunner_ODE_ss(rr_no_events, K_ref, 500)
    # y_test = results_test[1]
    # plt.plot(y_test)
    # plt.savefig('test2.png')

    # n_sobol_points = int(2**16)
    # p = sa.ProblemSpec({
    #     'names': ['k1_f', 'k1_r','k2_f', 'k2_r','k3_f', 'k3_r','k4_f', 'k4_r','k5_f', 'k5_r','k6_f', 'sigma'],
    #     'bounds':
    #         [ [6,12], [-1, 5], [-2, 4], [-2, 4], [3, 9], [-1, 5],
    #           [-1, 5], [6, 12], [-2, 4], [-2, 4], [-1, 5], [-11, -9],         
    #         ],
    # })

    # def wrapped_ssme(rr, n_points, K, y_true):
    #     y_pred= run_synthetic_ssme_assay(rr, n_points, K)[1]
    #     rmse = np.sqrt(np.mean(np.square(y_true-y_pred)))
    #     return rmse
    # print(wrapped_ssme(rr, 500, p_ref, y_true))
    # p.sample_sobol(n_sobol_points, calc_second_order=True)
    # param_values = p.samples

    bounds =[ [6,12], [-1, 5], [-2, 4], [-2, 4], [3, 9], [-1, 5],
              [-1, 5], [6, 12], [-2, 4], [-2, 4], [-1, 5], [-11, -9],    ]
    n_sobol_points = int(2**8)
    print(f'using {n_sobol_points} points')
    p = sa.ProblemSpec({
        'names': ['k1_f', 'k1_r','k2_f', 'k2_r','k3_f', 'k3_r','k4_f', 'k4_r','k5_f', 'k5_r','k6_f', 'sigma'],
        'bounds':
            [ [6,12], [-1, 5], [-2, 4], [-2, 4], [3, 9], [-1, 5],
              [-1, 5], [6, 12], [-2, 4], [-2, 4], [-1, 5], [-11, -9],         
            ],
    })
    p.sample_sobol(n_sobol_points, calc_second_order=True)
    P_sobol= p.samples
    n_iter = np.shape(P_sobol)[0]
    print(n_iter)

    max_logl_list = []    
    p_list = []
    t0 = time.time()

    p_init = [10.91836265,  3.44200738, -1.29832617,  2.46373216,  3.78882897,  1.41084257,
  3.05687311,  8.86143225, -1.87361744, -1.35843073, -0.46522219, -9.99256893]

    n_iter = 1e3
    for i in range(int(n_iter)):
        print(f'{i}/{n_iter}')
        #P_ref = P_sobol[i]
        #res = sp.optimize.minimize(nlogl_wrapper, P_ref, method='nelder-mead', args=(rr_no_events, y_ref), options={'disp': True})

        #res = sp.optimize.differential_evolution(nlogl_wrapper,bounds=bounds,args=(rr_no_events, y_ref), popsize=(2*12)+2, init='sobol', seed=i)
        res = sp.optimize.dual_annealing(nlogl_wrapper,bounds=bounds,args=(rr_no_events, y_ref), seed=i)
        p_list.append(res.x)
        max_logl_list.append(-1*res.fun)
    print(max_logl_list[-1])
    print(p_list[-1])
    print(P_ref)
    print(np.sqrt(np.mean(np.square(P_ref-np.array(p_list[-1])))))
    print(f'{time.time()-t0} s for {n_iter} iterations')
    np.savetxt('param_list_SA.csv', np.stack(p_list), delimiter=',')
    np.savetxt('max_logl_list_SA.csv', np.stack(max_logl_list), delimiter=',')
    #np.savetxt('max_logl.txt',max_logl_list)
    