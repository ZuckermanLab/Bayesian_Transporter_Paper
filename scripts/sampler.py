import numpy 
import scipy


def calc_normal_log_likelihood(y_obs, y_pred, sigma):
    '''calculates the log likelihood of a Normal distribution.'''
    return scipy.stats.norm.logpdf(y_obs, y_pred, sigma).sum()


def get_random_param_values(n_points, param_bounds, method='LHS', seed=0):

    
    N_dim = len(param_bounds)
    lower_bounds = [i[0] for i in param_bounds]
    upper_bounds = [i[1] for i in param_bounds]

    if method == 'LHS':
        sampler = scipy.stats.qmc.LatinHypercube(d=N_dim, seed= seed)
        samples = sampler.random(n=n_points)
        samples_scaled = scipy.stats.qmc.scale(samples, lower_bounds, upper_bounds)
        return samples_scaled
    else:
        raise NotImplementedError
    

def get_maximum_likelihood_estimate(objective_func, func_args, method, method_args, bounds, seed):
    
    if method == 'dual_annealing':
        res = scipy.optimize.dual_annealing(objective_func,bounds=bounds,args=func_args, seed=seed)
        return res
    if method == 'L-BFGS-B':
        x0 = method_args['x0']
        res = scipy.optimize.minimize(objective_func, x0, args=func_args, method='L-BFGS-B', bounds=bounds)
        return res
    if method == 'brute':
        res = scipy.optimize.brute(objective_func, bounds, Ns=2, args=func_args, full_output=True, finish=scipy.optimize.fmin)
        return res
   

