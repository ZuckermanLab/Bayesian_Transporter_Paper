# Readme

Supporting code for Bayesian Transporter research - still under development. 


### Usage:
for MLE use `run_optimizer.py`, for Bayesian use `run_emcee.py` or `run_pocomc.py`\

create an environment with required packages/dependencies using `conda env create -f environment.yml`

### Limitations / issues
- parallelization issues with libroadrunner. see: https://github.com/sys-bio/tellurium/issues/563 
- pocoMC sampler bugs, use my fork with fix. see: https://github.com/minaskar/pocomc/issues/22 and my fork: https://github.com/augeorge/pocomc 

### Docs:
- https://zuckermanlab.github.io/Bayesian_Transporter/ 

August George, 2023
