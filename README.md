# Readme

Supporting code for Bayesian Transporter research - still under development. 


### Usage:
create an environment with required packages/dependencies using `conda env create -f environment.yml` \

to run the provided example, update the filepaths in the `...config.yaml` file, and update the `...config.yaml` file path in `run_optimizer.py`, `run_emcee.py` and `run_pocomc.py`as needed.

for MLE with Scipy Optimization use `python YOUR_PATH_HERE/run_optimizer.py`, for Bayesian use `python YOUR_PATH_HERE/run_emcee.py` or `python YOUR_PATH_HERE/run_pocomc.py` to use the EMCEE or PocoMC packages respectively\


### Limitations / issues
- parallelization issues with libroadrunner. see: https://github.com/sys-bio/tellurium/issues/563 
- pocoMC sampler bugs, use my fork with fix. see: https://github.com/minaskar/pocomc/issues/22 and my fork: https://github.com/augeorge/pocomc 

### Docs:
- https://zuckermanlab.github.io/Bayesian_Transporter/ 

August George, 2023
