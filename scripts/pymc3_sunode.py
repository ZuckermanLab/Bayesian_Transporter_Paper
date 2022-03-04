
import numpy as np
import pymc3 as pm
from pymc3.ode import DifferentialEquation
import scipy
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
import theano 
import arviz as az
import sunode
import sunode.wrappers.as_theano
import sympy

# import multiprocessing as mp
# mp.set_start_method('fork')


def transporter_1c(y, t, theta, b):
    """ Single cycle model for a 1:1 secondary-active membrane transporter. 
    Assumes a fixed external concentration of the ion and substrate (e.g. experimental buffer solution)
    see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html 

    Args:
        y (numpy array): model state concentrations
        t (float): time point for integration
        theta (numpy array): model reaction rate constants
        b (numpy array): concentrations for the fixed external buffer solution (ion_out, substrate_out)

    Returns:
        (numpy array): integrated state concetrations at time point t (same order as y)
    """

    ### compartment size (in L)
    vol = 1
    v=1

    ### model parameters - reaction rate constants (in 1/s and 1/(Ms))
    rxn1_k1 = 0
    rxn1_k2 = 0
    rxn2_k1 = 10**theta[0]  # H on rate
    rxn2_k2 = 10**theta[1]  # H off rate
    rxn3_k1 = 10**theta[2]  # S off rate
    rxn3_k2 = 10**theta[3]  # S on rate
    rxn4_k1 = 10**theta[4]  # conf rate
    rxn4_k2 = 10**theta[5]  # conf rate
    rxn5_k1 = 0
    rxn5_k2 = 0
    rxn6_k1 = 10**theta[6]  # conf rate
    rxn6_k2 = 10**theta[7]  # conf rate
    rxn7_k1 = 0
    rxn7_k2 = 0
    rxn8_k1 = 0
    rxn8_k2 = 0
    rxn9_k1 = 0
    rxn9_k2 = 0
    rxn10_k1 = 0
    rxn10_k2 = 0
    rxn11_k1 = 10**theta[8]  # S on rate
    rxn11_k2 = 10**theta[9]  # S off rate
    rxn12_k1 = 10**theta[10]  # H off rate
    rxn12_k2 = (rxn2_k1*rxn3_k1*rxn4_k1*rxn6_k1*rxn11_k1*rxn12_k1)/(rxn2_k2*rxn3_k2*rxn4_k2*rxn6_k2*rxn11_k2)  # H on rate (cycle constraint)
  
    ### model state concentrations (in M)
    H_out = b[0]  # constant external ion concentration (e.g. buffer solution) 
    S_out = b[1]  # constant external substrate concentration (e.g. buffer solution) 
    
    H_in = y[0]  # internal ion concentration
    S_in = y[1]  # internal substrate concentation
    OF = y[2]  # fully unbound outward-facing transporter 
    IF = y[3]  # fully unbound inward-facing transporter
    OF_Hb = y[4]  # ion-bound only outward-facing transporter
    IF_Hb = y[5]  # ion-bound only inward-facing transporter
    OF_Sb = y[6]  # substrate-bound only outward-facing transporter
    IF_Sb = y[7]  # substrate-bound only inward-facing transporter
    OF_Hb_Sb = y[8]  # ion and substrate bound outward-facing transporter
    IF_Hb_Sb = y[9]  # ion and substrate bound inward-facing transporter

           
    ### reaction equations (from Tellurium)       
    vrxn1 = vol*(rxn1_k1*IF-rxn1_k2*OF)
    vrxn2 = vol*(rxn2_k1*OF*H_out-rxn2_k2*OF_Hb)
    vrxn3 = vol*(rxn3_k1*OF_Sb-rxn3_k2*OF*S_out)
    vrxn4 = vol*(rxn4_k1*OF_Hb-rxn4_k2*IF_Hb)
    vrxn5 = vol*(rxn5_k1*OF_Hb_Sb-rxn5_k2*OF_Hb*S_out)
    vrxn6 = vol*(rxn6_k1*IF_Sb-rxn6_k2*OF_Sb)
    vrxn7 = vol*(rxn7_k1*OF_Sb*H_out-rxn7_k2*OF_Hb_Sb)
    vrxn8 = vol*(rxn8_k1*OF_Hb_Sb-rxn8_k2*IF_Hb_Sb)
    vrxn9 = vol*(rxn9_k1*IF_Hb-rxn9_k2*IF*H_in)
    vrxn10 = vol*(rxn10_k1*IF*S_in-rxn10_k2*IF_Sb)
    vrxn11 = vol*(rxn11_k1*IF_Hb*S_in-rxn11_k2*IF_Hb_Sb)
    vrxn12 = vol*(rxn12_k1*IF_Hb_Sb-rxn12_k2*IF_Sb*H_in)

    ### ODE equations (from Tellurium)
    dOF_dt = v*(vrxn1 - vrxn2 + vrxn3)
    dOF_Hb_dt = v*(vrxn2 - vrxn4 + vrxn5)
    dIF_Hb_dt = v*(vrxn4 - vrxn9 - vrxn11)
    dS_in_dt = v*(-vrxn10 - vrxn11)
    dIF_Hb_Sb_dt = v*(vrxn8 + vrxn11 - vrxn12)
    dH_in_dt = v*(vrxn9 + vrxn12)
    dIF_Sb_dt = v*(-vrxn6 + vrxn10 + vrxn12)
    dOF_Sb_dt = v*(-vrxn3 + vrxn6 - vrxn7)
    dIF_dt = v*(-vrxn1 + vrxn9 - vrxn10)
    dOF_Hb_Sb_dt = v*(-vrxn5 + vrxn7 - vrxn8)
    ODE_list = [
        dH_in_dt,
        dS_in_dt,
        dOF_dt,
        dIF_dt,
        dOF_Hb_dt, 
        dIF_Hb_dt,
        dOF_Sb_dt,
        dIF_Sb_dt, 
        dOF_Hb_Sb_dt,
        dIF_Hb_Sb_dt,   
    ]
    return np.stack(ODE_list)


def transporter_s1(t, y, p):
    """ Single cycle model for a 1:1 secondary-active membrane transporter. 
    Assumes a fixed external concentration of the ion and substrate (e.g. experimental buffer solution)
    see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html 

    Args:
        y (numpy array): model state concentrations
        t (float): time point for integration
        theta (numpy array): model reaction rate constants
        b (numpy array): concentrations for the fixed external buffer solution (ion_out, substrate_out)

    Returns:
        (numpy array): integrated state concetrations at time point t (same order as y)
    """

    ### compartment size (in L)
    vol = 1
    v=1

    ### model parameters - reaction rate constants (in 1/s and 1/(Ms))
    rxn1_k1 = 0
    rxn1_k2 = 0
    rxn2_k1 = 10**p.log_rxn2_k1  # H on rate
    rxn2_k2 = 10**p.log_rxn2_k2  # H off rate
    rxn3_k1 = 10**p.log_rxn3_k1  # S off rate
    rxn3_k2 = 10**p.log_rxn3_k2   # S on rate
    rxn4_k1 = 10**p.log_rxn4_k1   # conf rate
    rxn4_k2 = 10**p.log_rxn4_k2   # conf rate
    rxn5_k1 = 0
    rxn5_k2 = 0
    rxn6_k1 = 10**p.log_rxn6_k1   # conf rate
    rxn6_k2 = 10**p.log_rxn6_k2   # conf rate
    rxn7_k1 = 0
    rxn7_k2 = 0
    rxn8_k1 = 0
    rxn8_k2 = 0
    rxn9_k1 = 0
    rxn9_k2 = 0
    rxn10_k1 = 0
    rxn10_k2 = 0
    rxn11_k1 = 10**p.log_rxn11_k1   # S on rate
    rxn11_k2 = 10**p.log_rxn11_k2   # S off rate
    rxn12_k1 = 10**p.log_rxn12_k1   # H off rate
    rxn12_k2 = (rxn2_k1*rxn3_k1*rxn4_k1*rxn6_k1*rxn11_k1*rxn12_k1)/(rxn2_k2*rxn3_k2*rxn4_k2*rxn6_k2*rxn11_k2)  # H on rate (cycle constraint)
  
    ### model state concentrations (in M)
    H_out = 1e-7  # constant external ion concentration (e.g. buffer solution) 
    S_out = 0.001  # constant external substrate concentration (e.g. buffer solution) 
    
    H_in = y.H_in  # internal ion concentration
    S_in = y.S_in  # internal substrate concentation
    OF = y.OF  # fully unbound outward-facing transporter 
    IF = y.IF  # fully unbound inward-facing transporter
    OF_Hb = y.OF_Hb  # ion-bound only outward-facing transporter
    IF_Hb = y.IF_Hb  # ion-bound only inward-facing transporter
    OF_Sb = y.OF_Sb  # substrate-bound only outward-facing transporter
    IF_Sb = y.IF_Sb  # substrate-bound only inward-facing transporter
    OF_Hb_Sb = y.OF_Hb_Sb  # ion and substrate bound outward-facing transporter
    IF_Hb_Sb = y.IF_Hb_Sb  # ion and substrate bound inward-facing transporter

           
    ### reaction equations (from Tellurium)       
    vrxn1 = vol*(rxn1_k1*IF-rxn1_k2*OF)
    vrxn2 = vol*(rxn2_k1*OF*H_out-rxn2_k2*OF_Hb)
    vrxn3 = vol*(rxn3_k1*OF_Sb-rxn3_k2*OF*S_out)
    vrxn4 = vol*(rxn4_k1*OF_Hb-rxn4_k2*IF_Hb)
    vrxn5 = vol*(rxn5_k1*OF_Hb_Sb-rxn5_k2*OF_Hb*S_out)
    vrxn6 = vol*(rxn6_k1*IF_Sb-rxn6_k2*OF_Sb)
    vrxn7 = vol*(rxn7_k1*OF_Sb*H_out-rxn7_k2*OF_Hb_Sb)
    vrxn8 = vol*(rxn8_k1*OF_Hb_Sb-rxn8_k2*IF_Hb_Sb)
    vrxn9 = vol*(rxn9_k1*IF_Hb-rxn9_k2*IF*H_in)
    vrxn10 = vol*(rxn10_k1*IF*S_in-rxn10_k2*IF_Sb)
    vrxn11 = vol*(rxn11_k1*IF_Hb*S_in-rxn11_k2*IF_Hb_Sb)
    vrxn12 = vol*(rxn12_k1*IF_Hb_Sb-rxn12_k2*IF_Sb*H_in)

    ### ODE equations (from Tellurium)
    dOF_dt = v*(vrxn1 - vrxn2 + vrxn3)
    dOF_Hb_dt = v*(vrxn2 - vrxn4 + vrxn5)
    dIF_Hb_dt = v*(vrxn4 - vrxn9 - vrxn11)
    dS_in_dt = v*(-vrxn10 - vrxn11)
    dIF_Hb_Sb_dt = v*(vrxn8 + vrxn11 - vrxn12)
    dH_in_dt = v*(vrxn9 + vrxn12)
    dIF_Sb_dt = v*(-vrxn6 + vrxn10 + vrxn12)
    dOF_Sb_dt = v*(-vrxn3 + vrxn6 - vrxn7)
    dIF_dt = v*(-vrxn1 + vrxn9 - vrxn10)
    dOF_Hb_Sb_dt = v*(-vrxn5 + vrxn7 - vrxn8)
    ODE_dict = {
        "H_in" : dH_in_dt,
        "S_in" : dS_in_dt,
        "OF": dOF_dt,
        "IF": dIF_dt,
        "OF_Hb": dOF_Hb_dt, 
        "IF_Hb": dIF_Hb_dt,
        "OF_Sb": dOF_Sb_dt,
        "IF_Sb": dIF_Sb_dt, 
        "OF_Hb_Sb": dOF_Hb_Sb_dt,
        "IF_Hb_Sb": dIF_Hb_Sb_dt,   
    }
    
    return ODE_dict

def transporter_s2(t, y, p):
    """ Single cycle model for a 1:1 secondary-active membrane transporter. 
    Assumes a fixed external concentration of the ion and substrate (e.g. experimental buffer solution)
    see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html 

    Args:
        y (numpy array): model state concentrations
        t (float): time point for integration
        theta (numpy array): model reaction rate constants
        b (numpy array): concentrations for the fixed external buffer solution (ion_out, substrate_out)

    Returns:
        (numpy array): integrated state concetrations at time point t (same order as y)
    """

    ### compartment size (in L)
    vol = 1
    v=1

    ### model parameters - reaction rate constants (in 1/s and 1/(Ms))
    rxn1_k1 = 0
    rxn1_k2 = 0
    rxn2_k1 = 10**p.log_rxn2_k1  # H on rate
    rxn2_k2 = 10**p.log_rxn2_k2  # H off rate
    rxn3_k1 = 10**p.log_rxn3_k1  # S off rate
    rxn3_k2 = 10**p.log_rxn3_k2   # S on rate
    rxn4_k1 = 10**p.log_rxn4_k1   # conf rate
    rxn4_k2 = 10**p.log_rxn4_k2   # conf rate
    rxn5_k1 = 0
    rxn5_k2 = 0
    rxn6_k1 = 10**p.log_rxn6_k1   # conf rate
    rxn6_k2 = 10**p.log_rxn6_k2   # conf rate
    rxn7_k1 = 0
    rxn7_k2 = 0
    rxn8_k1 = 0
    rxn8_k2 = 0
    rxn9_k1 = 0
    rxn9_k2 = 0
    rxn10_k1 = 0
    rxn10_k2 = 0
    rxn11_k1 = 10**p.log_rxn11_k1   # S on rate
    rxn11_k2 = 10**p.log_rxn11_k2   # S off rate
    rxn12_k1 = 10**p.log_rxn12_k1   # H off rate
    rxn12_k2 = (rxn2_k1*rxn3_k1*rxn4_k1*rxn6_k1*rxn11_k1*rxn12_k1)/(rxn2_k2*rxn3_k2*rxn4_k2*rxn6_k2*rxn11_k2)  # H on rate (cycle constraint)
  
    ### model state concentrations (in M)
    H_out = 5e-8  # constant external ion concentration (e.g. buffer solution) 
    S_out = 0.001  # constant external substrate concentration (e.g. buffer solution) 
    
    H_in = y.H_in  # internal ion concentration
    S_in = y.S_in  # internal substrate concentation
    OF = y.OF  # fully unbound outward-facing transporter 
    IF = y.IF  # fully unbound inward-facing transporter
    OF_Hb = y.OF_Hb  # ion-bound only outward-facing transporter
    IF_Hb = y.IF_Hb  # ion-bound only inward-facing transporter
    OF_Sb = y.OF_Sb  # substrate-bound only outward-facing transporter
    IF_Sb = y.IF_Sb  # substrate-bound only inward-facing transporter
    OF_Hb_Sb = y.OF_Hb_Sb  # ion and substrate bound outward-facing transporter
    IF_Hb_Sb = y.IF_Hb_Sb  # ion and substrate bound inward-facing transporter

           
    ### reaction equations (from Tellurium)       
    vrxn1 = vol*(rxn1_k1*IF-rxn1_k2*OF)
    vrxn2 = vol*(rxn2_k1*OF*H_out-rxn2_k2*OF_Hb)
    vrxn3 = vol*(rxn3_k1*OF_Sb-rxn3_k2*OF*S_out)
    vrxn4 = vol*(rxn4_k1*OF_Hb-rxn4_k2*IF_Hb)
    vrxn5 = vol*(rxn5_k1*OF_Hb_Sb-rxn5_k2*OF_Hb*S_out)
    vrxn6 = vol*(rxn6_k1*IF_Sb-rxn6_k2*OF_Sb)
    vrxn7 = vol*(rxn7_k1*OF_Sb*H_out-rxn7_k2*OF_Hb_Sb)
    vrxn8 = vol*(rxn8_k1*OF_Hb_Sb-rxn8_k2*IF_Hb_Sb)
    vrxn9 = vol*(rxn9_k1*IF_Hb-rxn9_k2*IF*H_in)
    vrxn10 = vol*(rxn10_k1*IF*S_in-rxn10_k2*IF_Sb)
    vrxn11 = vol*(rxn11_k1*IF_Hb*S_in-rxn11_k2*IF_Hb_Sb)
    vrxn12 = vol*(rxn12_k1*IF_Hb_Sb-rxn12_k2*IF_Sb*H_in)

    ### ODE equations (from Tellurium)
    dOF_dt = v*(vrxn1 - vrxn2 + vrxn3)
    dOF_Hb_dt = v*(vrxn2 - vrxn4 + vrxn5)
    dIF_Hb_dt = v*(vrxn4 - vrxn9 - vrxn11)
    dS_in_dt = v*(-vrxn10 - vrxn11)
    dIF_Hb_Sb_dt = v*(vrxn8 + vrxn11 - vrxn12)
    dH_in_dt = v*(vrxn9 + vrxn12)
    dIF_Sb_dt = v*(-vrxn6 + vrxn10 + vrxn12)
    dOF_Sb_dt = v*(-vrxn3 + vrxn6 - vrxn7)
    dIF_dt = v*(-vrxn1 + vrxn9 - vrxn10)
    dOF_Hb_Sb_dt = v*(-vrxn5 + vrxn7 - vrxn8)
    ODE_dict = {
        "H_in" : dH_in_dt,
        "S_in" : dS_in_dt,
        "OF": dOF_dt,
        "IF": dIF_dt,
        "OF_Hb": dOF_Hb_dt, 
        "IF_Hb": dIF_Hb_dt,
        "OF_Sb": dOF_Sb_dt,
        "IF_Sb": dIF_Sb_dt, 
        "OF_Hb_Sb": dOF_Hb_Sb_dt,
        "IF_Hb_Sb": dIF_Hb_Sb_dt,   
    }
    
    return ODE_dict

def transporter_st(t, y, p):
    """ Single cycle model for a 1:1 secondary-active membrane transporter. 
    Assumes a fixed external concentration of the ion and substrate (e.g. experimental buffer solution)
    see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html 

    Args:
        y (numpy array): model state concentrations
        t (float): time point for integration
        theta (numpy array): model reaction rate constants
        b (numpy array): concentrations for the fixed external buffer solution (ion_out, substrate_out)

    Returns:
        (numpy array): integrated state concetrations at time point t (same order as y)
    """

    ### compartment size (in L)
    vol = 1
    v=1

    ### model parameters - reaction rate constants (in 1/s and 1/(Ms))
    rxn1_k1 = 0
    rxn1_k2 = 0
    rxn2_k1 = 10**p.log_rxn2_k1  # H on rate
    rxn2_k2 = 10**p.log_rxn2_k2  # H off rate
    rxn3_k1 = 10**p.log_rxn3_k1  # S off rate
    rxn3_k2 = 10**p.log_rxn3_k2   # S on rate
    rxn4_k1 = 10**p.log_rxn4_k1   # conf rate
    rxn4_k2 = 10**p.log_rxn4_k2   # conf rate
    rxn5_k1 = 0
    rxn5_k2 = 0
    rxn6_k1 = 10**p.log_rxn6_k1   # conf rate
    rxn6_k2 = 10**p.log_rxn6_k2   # conf rate
    rxn7_k1 = 0
    rxn7_k2 = 0
    rxn8_k1 = 0
    rxn8_k2 = 0
    rxn9_k1 = 0
    rxn9_k2 = 0
    rxn10_k1 = 0
    rxn10_k2 = 0
    rxn11_k1 = 10**p.log_rxn11_k1   # S on rate
    rxn11_k2 = 10**p.log_rxn11_k2   # S off rate
    rxn12_k1 = 10**p.log_rxn12_k1   # H off rate
    rxn12_k2 = (rxn2_k1*rxn3_k1*rxn4_k1*rxn6_k1*rxn11_k1*rxn12_k1)/(rxn2_k2*rxn3_k2*rxn4_k2*rxn6_k2*rxn11_k2)  # H on rate (cycle constraint)
  
    ### model state concentrations (in M)
    
    expr = t-5
    if (expr).is_positive:
        H_out = 1e-7
        print(expr) 
        print('using 1e-7')
    else:
        H_out = 5e-8  # constant external ion concentration (e.g. buffer solution)
        print(expr) 
        print('using 5e-8')
       
    S_out = 0.001  # constant external substrate concentration (e.g. buffer solution) 
    
    H_in = y.H_in  # internal ion concentration
    S_in = y.S_in  # internal substrate concentation
    OF = y.OF  # fully unbound outward-facing transporter 
    IF = y.IF  # fully unbound inward-facing transporter
    OF_Hb = y.OF_Hb  # ion-bound only outward-facing transporter
    IF_Hb = y.IF_Hb  # ion-bound only inward-facing transporter
    OF_Sb = y.OF_Sb  # substrate-bound only outward-facing transporter
    IF_Sb = y.IF_Sb  # substrate-bound only inward-facing transporter
    OF_Hb_Sb = y.OF_Hb_Sb  # ion and substrate bound outward-facing transporter
    IF_Hb_Sb = y.IF_Hb_Sb  # ion and substrate bound inward-facing transporter

           
    ### reaction equations (from Tellurium)       
    vrxn1 = vol*(rxn1_k1*IF-rxn1_k2*OF)
    vrxn2 = vol*(rxn2_k1*OF*H_out-rxn2_k2*OF_Hb)
    vrxn3 = vol*(rxn3_k1*OF_Sb-rxn3_k2*OF*S_out)
    vrxn4 = vol*(rxn4_k1*OF_Hb-rxn4_k2*IF_Hb)
    vrxn5 = vol*(rxn5_k1*OF_Hb_Sb-rxn5_k2*OF_Hb*S_out)
    vrxn6 = vol*(rxn6_k1*IF_Sb-rxn6_k2*OF_Sb)
    vrxn7 = vol*(rxn7_k1*OF_Sb*H_out-rxn7_k2*OF_Hb_Sb)
    vrxn8 = vol*(rxn8_k1*OF_Hb_Sb-rxn8_k2*IF_Hb_Sb)
    vrxn9 = vol*(rxn9_k1*IF_Hb-rxn9_k2*IF*H_in)
    vrxn10 = vol*(rxn10_k1*IF*S_in-rxn10_k2*IF_Sb)
    vrxn11 = vol*(rxn11_k1*IF_Hb*S_in-rxn11_k2*IF_Hb_Sb)
    vrxn12 = vol*(rxn12_k1*IF_Hb_Sb-rxn12_k2*IF_Sb*H_in)

    ### ODE equations (from Tellurium)
    dOF_dt = v*(vrxn1 - vrxn2 + vrxn3)
    dOF_Hb_dt = v*(vrxn2 - vrxn4 + vrxn5)
    dIF_Hb_dt = v*(vrxn4 - vrxn9 - vrxn11)
    dS_in_dt = v*(-vrxn10 - vrxn11)
    dIF_Hb_Sb_dt = v*(vrxn8 + vrxn11 - vrxn12)
    dH_in_dt = v*(vrxn9 + vrxn12)
    dIF_Sb_dt = v*(-vrxn6 + vrxn10 + vrxn12)
    dOF_Sb_dt = v*(-vrxn3 + vrxn6 - vrxn7)
    dIF_dt = v*(-vrxn1 + vrxn9 - vrxn10)
    dOF_Hb_Sb_dt = v*(-vrxn5 + vrxn7 - vrxn8)
    ODE_dict = {
        "H_in" : dH_in_dt,
        "S_in" : dS_in_dt,
        "OF": dOF_dt,
        "IF": dIF_dt,
        "OF_Hb": dOF_Hb_dt, 
        "IF_Hb": dIF_Hb_dt,
        "OF_Sb": dOF_Sb_dt,
        "IF_Sb": dIF_Sb_dt, 
        "OF_Hb_Sb": dOF_Hb_Sb_dt,
        "IF_Hb_Sb": dIF_Hb_Sb_dt,   
    }
    
    return ODE_dict

np.random.seed(1)

### time steps
sample_rate = 25  # sample rate (x samples/sec)
stage_time = 5 # how long each experiment stage is (in sec)
sample_rate = 25  # sample rate (x samples/sec)
stage_time = 5 # how long each experiment stage is (in sec)
t1 = np.linspace(0,stage_time,int(sample_rate*stage_time), endpoint=False)
t2 = np.linspace(stage_time,2*stage_time,int(sample_rate*stage_time), endpoint=False)
t3 = np.linspace(2*stage_time,3*stage_time,int(sample_rate*stage_time),endpoint=False)
t_tot = np.hstack([t1,t2,t3])
t_23 = np.hstack([t2,t3])
t_s1_and_s2 = np.hstack([t1,t2])

print(t2)

### initial concentrations
y0 = np.array([
    1e-7,
    1e-3,
    2.833e-08,
    2.125e-08,
    2.833e-08,
    2.833e-08,
    2.125e-08,
    2.125e-08,
    2.125e-08,
    2.833e-08,
])

### experimental buffer concentrations
b1 = np.array([1e-07, 0.001])
b2 = np.array([5e-08, 0.001])
b3 = np.array([1e-07, 0.001])

### rate constants
k_H_on = 1e10
k_H_off = 1e3
k_S_on = 1e7
k_S_off = 1e3
k_conf = 1e2
theta = np.array([
    np.log10(k_H_on),
    np.log10(k_H_off),
    np.log10(k_S_off),
    np.log10(k_S_on),
    np.log10(k_conf),
    np.log10(k_conf),
    np.log10(k_conf),
    np.log10(k_conf),
    np.log10(k_S_on),  
    np.log10(k_S_off),  
    np.log10(k_H_off),  
])
std_true = 1e-13

### integrate multiple experiments
s1 = odeint(transporter_1c, y0=y0, t=t1, args=(theta,b1), atol=1e-16, rtol=1e-14)
s2 = odeint(transporter_1c, y0=s1[-1],t=t2, args=(theta,b2), atol=1e-16, rtol=1e-14)
s3 = odeint(transporter_1c, y0=s2[-1],t=t3, args=(theta,b3), atol=1e-16, rtol=1e-14)
# #D = np.transpose(np.vstack([s2,s3]))


D = np.transpose(np.vstack([s1,s2]))
#D_s1 = np.transpose(np.vstack([s1]))
# vol*(rxn12_k1*IF_Hb_Sb-rxn12_k2*IF_Sb*H_in)
y_true = np.array((0.0001*(1e3*D[9]-1e10*D[7]*D[0])))
#y_true2 = np.array((0.0001*(1e3*D_s1[9]-1e10*D_s1[7]*D_s1[0])))

noise = np.random.normal(0,std_true, np.size(y_true))
y_obs = y_true + noise 
# plt.figure(figsize=(15,10))
# # plt.plot(t_23, y_true)
# # plt.plot(t_23, y_true2, '--')
# # plt.plot(t_23, y_obs, 'o')
# plt.plot(t_s1_and_s2, y_true)
# plt.plot(t_s1_and_s2, y_obs, 'o')
# #plt.plot(t1, y_true2)

seed = 1

with pm.Model() as model:
    np.random.seed(seed)
    stdev = pm.Uniform("noise_stdev", std_true*0.25, std_true*1.50)
    s=1  # shift priors so the true value isn't centered
    log_rxn2_k1 = pm.Uniform("log_rxn2_k1", 7-s, 13-s, testval=np.random.uniform(7-s, 13-s))
    log_rxn2_k2 = pm.Uniform("log_rxn2_k2", 0-s, 6-s, testval=np.random.uniform(0-s, 6-s))
    log_rxn3_k1 = pm.Uniform("log_rxn3_k1", 0-s, 6-s, testval=np.random.uniform(0-s, 6-s))
    log_rxn3_k2 = pm.Uniform("log_rxn3_k2", 4-s, 10-s, testval=np.random.uniform(4-s, 10-s))
    log_rxn4_k1 = pm.Uniform("log_rxn4_k1", -1-s, 5-s, testval=np.random.uniform(-1-s, 5-s))
    log_rxn4_k2 = pm.Uniform("log_rxn4_k2", -1-s, 5-s, testval=np.random.uniform(-1-s, 5-s))
    log_rxn6_k1 = pm.Uniform("log_rxn6_k1", -1-s, 5-s, testval=np.random.uniform(-1-s, 5-s))
    log_rxn6_k2 = pm.Uniform("log_rxn6_k2", -1-s, 5-s, testval=np.random.uniform(-1-s, 5-s))
    log_rxn11_k1 = pm.Uniform("log_rxn11_k1", 4-s, 10-s, testval=np.random.uniform(4-s, 10-s))
    log_rxn11_k2 = pm.Uniform("log_rxn11_k2", 0-s, 6-s, testval=np.random.uniform(0-s, 6-s))
    log_rxn12_k1 = pm.Uniform("log_rxn12_k1", 0-s, 6-s, testval=np.random.uniform(0-s, 6-s))
    # log_rxn12_k2 = pm.Deterministic('log_rxn12_k2', ((log_rxn2_k1+log_rxn3_k1+log_rxn4_k1+log_rxn6_k1+log_rxn11_k1+log_rxn12_k1)-(log_rxn2_k2+log_rxn3_k2+log_rxn4_k2+log_rxn6_k2+log_rxn11_k2)))
    log_rxn12_k2 = ((log_rxn2_k1+log_rxn3_k1+log_rxn4_k1+log_rxn6_k1+log_rxn11_k1+log_rxn12_k1)-(log_rxn2_k2+log_rxn3_k2+log_rxn4_k2+log_rxn6_k2+log_rxn11_k2))
    theta_sample = [
        log_rxn2_k1, 
        log_rxn2_k2,
        log_rxn3_k1, 
        log_rxn3_k2,
        log_rxn4_k1, 
        log_rxn4_k2,
        log_rxn6_k1, 
        log_rxn6_k2,
        log_rxn11_k1, 
        log_rxn11_k2,
        log_rxn12_k1, 
    ]
    y_hat, _, problem, solver, _, _ = sunode.wrappers.as_theano.solve_ivp(
        y0={
        # The initial conditions of the ode. Each variable
        # needs to specify a theano or numpy variable and a shape.
        # This dict can be nested.
            "H_in" : (np.array(y0[0]), ()),
            "S_in" : (np.array(y0[1]), ()),
            "OF": (np.array(y0[2]), ()),
            "IF": (np.array(y0[3]), ()),
            "OF_Hb": (np.array(y0[4]), ()), 
            "IF_Hb": (np.array(y0[5]), ()),
            "OF_Sb": (np.array(y0[6]), ()),
            "IF_Sb": (np.array(y0[7]), ()), 
            "OF_Hb_Sb": (np.array(y0[8]), ()),
            "IF_Hb_Sb": (np.array(y0[9]), ()),   
        },
        params={
        # Each parameter of the ode. sunode will only compute derivatives
        # with respect to theano variables. The shape needs to be specified
        # as well. It it infered automatically for numpy variables.
        # This dict can be nested.
            'log_rxn2_k1': (log_rxn2_k1, ()),
            'log_rxn2_k2': (log_rxn2_k2, ()),
            'log_rxn3_k1': (log_rxn3_k1, ()),
            'log_rxn3_k2': (log_rxn3_k2, ()),
            'log_rxn4_k1': (log_rxn4_k1, ()),
            'log_rxn4_k2': (log_rxn4_k2, ()),
            'log_rxn6_k1': (log_rxn6_k1, ()),
            'log_rxn6_k2': (log_rxn6_k2, ()),
            'log_rxn11_k1': (log_rxn11_k1, ()),
            'log_rxn11_k2': (log_rxn11_k2, ()),
            'log_rxn12_k1': (log_rxn12_k1, ()),
            'extra': np.zeros(1),
        },
        # A functions that computes the right-hand-side of the ode using
        # sympy variables.
        rhs=transporter_s1,
        # The time points where we want to access the solution
        tvals=t1,
        t0=t1[0],
    )
    
    y_hat2, _, problem2, solver2, _, _ = sunode.wrappers.as_theano.solve_ivp(
        y0={
        # The initial conditions of the ode. Each variable
        # needs to specify a theano or numpy variable and a shape.
        # This dict can be nested.
            "H_in" : (y_hat['H_in'][-1], ()),
            "S_in" : (y_hat['S_in'][-1], ()),
            "OF": (y_hat['OF'][-1], ()),
            "IF": (y_hat['IF'][-1], ()),
            "OF_Hb": (y_hat['OF_Hb'][-1], ()), 
            "IF_Hb": (y_hat['IF_Hb'][-1], ()),
            "OF_Sb": (y_hat['OF_Sb'][-1], ()),
            "IF_Sb": (y_hat['IF_Sb'][-1], ()), 
            "OF_Hb_Sb": (y_hat['OF_Hb_Sb'][-1], ()),
            "IF_Hb_Sb": (y_hat['IF_Hb_Sb'][-1], ()),   
        },
        params={
        # Each parameter of the ode. sunode will only compute derivatives
        # with respect to theano variables. The shape needs to be specified
        # as well. It it infered automatically for numpy variables.
        # This dict can be nested.
            'log_rxn2_k1': (log_rxn2_k1, ()),
            'log_rxn2_k2': (log_rxn2_k2, ()),
            'log_rxn3_k1': (log_rxn3_k1, ()),
            'log_rxn3_k2': (log_rxn3_k2, ()),
            'log_rxn4_k1': (log_rxn4_k1, ()),
            'log_rxn4_k2': (log_rxn4_k2, ()),
            'log_rxn6_k1': (log_rxn6_k1, ()),
            'log_rxn6_k2': (log_rxn6_k2, ()),
            'log_rxn11_k1': (log_rxn11_k1, ()),
            'log_rxn11_k2': (log_rxn11_k2, ()),
            'log_rxn12_k1': (log_rxn12_k1, ()),
            'extra': np.zeros(1),
        },
        # A functions that computes the right-hand-side of the ode using
        # sympy variables.
        rhs=transporter_s2,
        # The time points where we want to access the solution
        tvals=t2,
        t0=t2[0],
    )

    lib = sunode._cvodes.lib
    lib.CVodeSStolerances(solver._ode, 1e-16, 1e-18)  # rel tol, abs tol
    lib.CVodeSStolerances(solver2._ode, 1e-16, 1e-18)  # rel tol, abs tol
    lib.CVodeSetMaxNumSteps(solver._ode, 1000)
    lib.CVodeSetMaxNumStepsB(solver._ode, solver._odeB, 1000)
    lib.CVodeSetMaxNumSteps(solver2._ode, 1000)
    lib.CVodeSetMaxNumStepsB(solver2._ode, solver._odeB, 1000)
    
    
    IF_Hb_Sb_sample = pm.math.concatenate([y_hat['IF_Hb_Sb'],y_hat2['IF_Hb_Sb']])
    IF_Sb_sample = pm.math.concatenate([y_hat['IF_Sb'],y_hat2['IF_Sb']])
    H_in_sample = pm.math.concatenate([y_hat['H_in'],y_hat2['H_in']])

    # vol*(rxn12_k1*IF_Hb_Sb-rxn12_k2*IF_Sb*H_in)
    vol = 0.0001
    y_pred = vol*(((10**log_rxn12_k1)*IF_Hb_Sb_sample) - ((10**log_rxn12_k2)*IF_Sb_sample*H_in_sample))

    Y = pm.Normal('y_obs', mu=y_pred, sd=stdev, observed=y_obs)


    trace = pm.sample(tune=200, draws=800, chains=1, cores=1, random_seed=seed)
data = az.from_pymc3(trace=trace)
fig1 = az.plot_trace(data)
plt.savefig('fig1.png')
fig2 = az.plot_posterior(data)
plt.savefig('fig2.png')
print(az.summary(data).round(3))