import numpy as np
#import seaborn as sns
import pandas as pd
import arviz as az



class MCMCAnalysis:
    def __init__(self, sample_file):
        self.sample_file = sample_file
        self.samples = np.loadtxt(self.sample_file, delimiter=',')

    def r_hat(self, data):
        inf_data = az.convert_to_dataset(data)
        return az.rhat(inf_data)

    def plot_corner(self):
        # Plot 1D and 2D marginal distributions using seaborn or ArviZ's plot_pair function
        return sns.pairplot(pd.DataFrame(self.samples))
        
