import pypesto
import pypesto.optimize as optimize
import pypesto.sample as sampling
from scipy.optimize import minimize

class OptimizerSampler:
    def __init__(self, method, method_args):
        self.method = method
        self.method_args = method_args

    def optimize(self, problem, x0=None, **kwargs):
        if self.method in pypesto.optimize.__all__:
            optimizer_class = getattr(optimize, self.method)
            optimizer_instance = optimizer_class(**self.method_kwargs)
            result = optimize.minimize(problem, x0=x0, optimizer=optimizer_instance)
            x_opt = result.optimize_result.list[0]['x']
        elif self.method == 'scipy_minimize':
            result = minimize(problem.objective, x0, **self.method_kwargs)
            x_opt = result.x
        else:
            raise ValueError(f"Unsupported optimizer '{self.method}'.")
        return x_opt

    def sample(self, problem, n_samples, x0=None, **kwargs):
        if self.method in pypesto.sampling.__all__:
            sampler_class = getattr(sampling, self.method)
            sampler_instance = sampler_class(**self.method_kwargs)
            result = sampling.sample(problem, n_samples, sampler=sampler_instance, x0=x0)
            samples = result.sample_result.trace_x
        elif self.method.lower() == 'pocomc':
            samples = self.run_poco_mc(problem, n_samples, x0)
        else:
            raise ValueError(f"Unsupported sampler '{self.method}'.")
        return samples

    def run_poco_mc(self, problem, n_samples, x0):
        # Implement your PocoMC sampling code here
        # You should return a numpy array with shape (n_samples, problem.dim)
        pass
