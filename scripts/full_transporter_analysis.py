import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import corner 

f = '/Users/georgeau/Desktop/GitHub/august/model_identification/affine_MCMC_PT/full_transporter_data_v6.csv'
n_cols = 25
D_list = []
col_list = [i+1 for i in range(n_cols)]  # only keep columns 1...n_p-1 
D_tmp = np.genfromtxt(f, delimiter=',', skip_header=1,usecols=col_list).T  
print('data loaded')

exit()
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


fig = corner.corner(
        D_tmp, labels=labels)
plt.savefig(f'test_2dcorr.png')