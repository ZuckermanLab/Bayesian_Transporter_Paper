# Readme

Supporting code for Bayesian Transporter research - still under development. 


### Getting started
- create an environment with required packages/dependencies using `conda env create -f environment.yml` 
- run through the `example_notebook.ipynb` notebook for example usage and analysis
    - ensure that the filepaths are correctly set for your system in the configuration file `...config.yaml` configuration file.
- an `\example` directory is included in the repo which contains data from a short `pocomc` sampling run, along with an example configuration file, SBML and tellurium transporter model, and synthetic observed dataset.


### Stand alone usage
- For Bayesian inference using EMCEE run `python YOUR_PATH_HERE/run_emcee.py` in a terminal
- For Bayesian inference using PocoMC run `python YOUR_PATH_HERE/run_pocomc.py` in a terminal
- For Maximum Likelihood Estimation using Scipy Optimization run `python YOUR_PATH_HERE/run_optimizer.py.py` in a terminal

Ensure that the configuration file and file path is updated for each module as needed


### Limitations / issues
- parallelization issues with libroadrunner. see: https://github.com/sys-bio/tellurium/issues/563 
- pocoMC sampler bugs, use my fork with fix. see: https://github.com/minaskar/pocomc/issues/22 and my fork: https://github.com/augeorge/pocomc 

### Docs:
More detailed API documentation is provided below:
- https://zuckermanlab.github.io/Bayesian_Transporter/ 

August George, 2023
