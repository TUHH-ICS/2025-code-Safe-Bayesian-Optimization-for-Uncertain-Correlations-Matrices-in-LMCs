from utils.functions import MixedMTRKHSFunction
from botorch.utils.sampling import draw_sobol_samples
from utils.utils import unnormalize, normalize
from gpytorch.kernels import RBFKernel
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

"""
Generates and saves initial starting points for optimization tasks.

This script is designed to find suitable initial input points (`X_init`) for a
specified multi-task objective function (`MixedMTRKHSFunction`). It iterates
through a predefined list of random seeds to generate multiple starting points
for reproducibility.

For each seed, the script performs the following steps:
1.  Initializes a 4-dimensional `MixedMTRKHSFunction` with specific kernel
    parameters.
2.  Generates a large set of candidate points (1024) using Sobol sampling
    within the function's normalized bounds.
3.  Evaluates the objective function for these candidate points.
4.  Filters the points, keeping only those whose function output falls within a
    predefined range relative to a threshold `T`.
5.  Randomly selects one point from the filtered set to serve as the initial
    input `X_init`.
6.  Saves a dictionary containing the selected `X_init` and the `threshold` `T`
    to a PyTorch file (`.pt`) in the `data/x_init_dim4/` directory. The
    filename is formatted based on the function name and the seed.
"""

def get_init_inputs(bounds, dim=4, seeds=None):
    if seeds is None:
        seeds = [73, 1, 92, 23, 45, 67, 89, 12, 34, 56, 78, 90, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888]
        
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        kernel = RBFKernel(ard_num_dims=dim)
        kernel.lengthscale = torch.ones(dim)*.2
        norm_bounds = torch.tensor([[-1.0], [1.0]]).repeat(1,dim)
        obj = MixedMTRKHSFunction(cor=[.85,.0], B=[30.5,10.2],id_norm=True, only_task_2=False, bounds=norm_bounds, kernel=kernel) 
        T=-1.


        bounds = obj.bounds
        norm_bound = normalize(bounds,bounds)

        c = 0
        while c <1:
            A = draw_sobol_samples(norm_bound,1024,1).squeeze().to(dtype=torch.float64)

            vals = obj.f(A,0)
            vals2 = obj.f(A,1)
            ids = [(vals.squeeze()>=1.15*T) & (vals.squeeze()<=1.3*T)] if T>0 else [(vals.squeeze()>=.55*T) & (vals.squeeze()<=.0*T)]
            X_init = unnormalize(A[ids],bounds)[torch.randint(low=0,high=A[ids].size(0),size=(1,))]
            c = X_init.shape[0]
        data = {"X_init":X_init, "threshold":T}
        return data