import numpy as np
from scipy.optimize import root


##would love to remove this one day
Kb = 0.001987
T = 237.15 + 25
V0 = 1.42e-3


#make list of injections
def get_injlist(injno,indiv_vol):
    """<placeholder docstring. we should add a short description as well as list the inputs and outputs.>"""
    injvol = [indiv_vol for i in range(injno)]
    return injvol

##function for root finding
def find_p_l(variables,pt,lt,k1,k2):
    """<placeholder docstring. we should add a short description as well as list the inputs and outputs.>"""
    p,l = variables
    
    equation_1 = pt - ((p) + ((2*l*p)/ k1 ) + (2*p*p*l) / (k1*k2))
    equation_2 = lt - ((l) + ((2*l*p)/ k1 ) + (p*p*l) / (k1*k2)) 

    return abs(equation_1),abs(equation_2)
        
##take volumes, initial concs (P0, Ls in bayesitc terms), return list of total concentrations
##this is stripped more or less directly from bayesitc but I've tested it and it is correct
def get_simulate_tots(V0,injvol,syringe_conc,cell_conc):
    """<placeholder docstring. we should add a short description as well as list the inputs and outputs.>"""

    injecting_tot = [0]
    cell_tot = [cell_conc]
    
    ##cumulative dilution factor
    dcum = 1  
    
    #calculate totals for each inj
    for inj in injvol:
        d  = 1 - (inj/V0)
        dcum *= d
        P = syringe_conc * (1-dcum)
        L = cell_conc * dcum
        
        ##change to numpy arrays when you get a chance
        injecting_tot.append(P)
        cell_tot.append(L)
    return injecting_tot,cell_tot


def get_dq_list(dg,ddg,dh,ddh,pt,lt,dh_0,inj_list):
    """<placeholder docstring. we should add a short description as well as list the inputs and outputs.>"""
    k1 = np.exp(dg/(Kb*T)) 
    k2 = np.exp((dg+ddg)/(Kb*T))
    
    pt_list,lt_list = get_simulate_tots(V0,inj_list,pt,lt)

    
    q_list = []
    for i in range(len(pt_list)):
        pt = pt_list[i]
        lt = lt_list[i]
        
        
        #if i < 3:
        sol = root(find_p_l,method='lm',x0=(1e-8,1e-8),args=(pt,lt,k1,k2),options={'ftol':1e-20})
        #else:
        #    sol = root(find_p_l,method='lm',x0=(sol.x),args=(pt,lt,k1,k2),options={'ftol':1e-20})
        #print(sol.x)
        pfree = sol.x[0]
        lfree = sol.x[1]
        root_check = pt - pfree - ((pfree*lfree*2)/k1) - ((2*pfree*pfree*lfree)/(k1*k2))
        if root_check > 1e-10:
            print('Uh Oh... could not find root')
            print(dg,ddg,dh,ddh,i)
            print(sol.x)
        
        pl = 2*pfree*lfree / k1
        pl2 = lfree*pfree**2 / (k1*k2)
        q = V0 * (dh*pl + ((dh+(dh+ddh))*pl2))
        q_list.append(q)
        
    dq_list = np.zeros(len(q_list)-1)
    for i in range(1,len(q_list)-1):
        dq = q_list[i+1] - q_list[i] + inj_list[i] / V0 * ((q_list[i+1] + q_list[i])/2)
        ##unit conversion from kcal to ucal (dh_0 in ucal already)
        dq_list[i] = dq*1e9
    return dq_list




    ##trim the dq list. Slower than trimming at above step but the dq calculation is minimal in any case
    #adjusted_list = []
    #for i in range(len(dq_list)):
    #    if i in included_point_list:
    #        adjusted_list.append(dq_list[i])
    #print(dq_list,adjusted_list)
    #return adjusted_list


##alternate dq_list func that takes a list of included points
def get_dq_list_shortened(dg,dh,ddg,ddh,pt,lt,dh_0,inj_list,included_list):
    """<placeholder docstring. we should add a short description as well as list the inputs and outputs.>"""
    k1 = np.exp(dg/(Kb*T)) 
    k2 = np.exp((dg+ddg)/(Kb*T))
    
    pt_list,lt_list = get_simulate_tots(V0,inj_list,pt,lt)

    
    q_list = []

    ##clumsy way to ensure only needed states are calculated
    qset = set()
    for point in included_list:
        qset.add(point)
        qset.add(point+1)
    needed_q = list(qset)
  
    for i in range(len(pt_list)):
        pt = pt_list[i]
        lt = lt_list[i]
        
        
        if i in needed_q:
            sol = root(find_p_l,method='lm',x0=(1e-8,1e-8),args=(pt,lt,k1,k2),options={'ftol':1e-20})

            pfree = sol.x[0]
            lfree = sol.x[1]
            root_check = pt - pfree - ((pfree*lfree*2)/k1) - ((2*pfree*pfree*lfree)/(k1*k2))
            if root_check > 1e-9:
                print('Uh Oh... could not find root')
                print(dg,ddg,dh,ddh,i,root_check)
                print(sol.x)
                
            pl = 2*pfree*lfree / k1
            pl2 = lfree*pfree**2 / (k1*k2)
            q = V0 * (dh*pl + ((dh+(dh+ddh))*pl2))
            q_list.append(q)
        else:
            q_list.append(0)
    
    dq_list = np.zeros(len(q_list)-1)
    for i in range(1,len(q_list)-1):
        dq = q_list[i+1] - q_list[i] + inj_list[i] / V0 * ((q_list[i+1] + q_list[i])/2)
        ##unit conversion from kcal to ucal (dh_0 in ucal already)
        dq_list[i] = dq*1e9 + dh_0
 
    #print(dh,ddh,dq_list)
    modified_dq_list = np.zeros(len(included_list))
    for i in range(len(included_list)):
        modified_dq_list[i] = dq_list[included_list[i]]
    #print(modified_dq_list)        
    return modified_dq_list



def get_synthetic_itc(seed,conc_priors):
    """<placeholder docstring. we should add a short description as well as list the inputs and outputs.>"""
    
    np.random.seed(seed)
    
    
    ##itc constants and data synth block
    Kb = 0.001987
    T = 237.15 + 25
    V0 = 1.42e-3
    pt_stated = 500e-6
    lt_stated = 17e-6
    
    #kcal units
    dg = -7
    dh = -10
    ddg = -1
    ddh = -1.5
    dh_0 = 0
    
    #ucal units
    sigma = 0.2
    
    if conc_priors:
        theta_true = [dg,dh,ddg,ddh,pt_stated,lt_stated,sigma]
    else:
        theta_true = [dg,dh,ddg,ddh,sigma]
    inj_list = []
    
    
    #build injection list
    #must be converted to L before use, currently in uL.
    inj_count = 35
    inj_vol = 6
    inj_list.append(2)
    for i in range(inj_count):
        inj_list.append(inj_vol)
        
    ##conversion to liter values
    inj_list_l = [inj_list[i]*1e-6 for i in inj_list]
    
    #total concentrations per injection
    ptot,ltot = get_simulate_tots(V0,inj_list_l,pt_stated,lt_stated)

    #dqs
    true_dq = get_dq_list(dg,ddg,dh,ddh,pt_stated,lt_stated,dh_0,inj_list_l)
    dq_obs = true_dq + np.random.normal(loc=0,scale=sigma,size=np.size(true_dq))
    print(f'true_dq: {true_dq}')
    
    if conc_priors:
        return true_dq, dq_obs,theta_true,[inj_list_l]
    else:
        return true_dq, dq_obs,theta_true,[inj_list_l,pt_stated,lt_stated]
