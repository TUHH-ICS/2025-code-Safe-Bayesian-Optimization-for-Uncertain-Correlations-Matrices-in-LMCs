#!/usr/bin/env python3

"""
This script runs a single-task Bayesian Optimization experiment on a multi-task benchmark function.

The script performs the following steps:
- Initializes parameters for the optimization, including dimensionality, random seeds, and function settings.
- Iterates through a list of random seeds to run multiple independent trials.
- For each trial:
    - Sets up a multi-task objective function (`MixedMTRKHSFunction`).
    - Gathers initial data by evaluating a starting point on all tasks and sampling supplementary data from other tasks.
    - Initializes a `BayesianOptimization` instance configured for single-task optimization on the primary task.
    - Builds an initial multi-task Gaussian Process (GP) model using an Intrinsic Coregionalization Model (ICM).
    - Runs the main optimization loop for a specified number of iterations (`nruns`). In each step, it updates the GP model and performs one step of Bayesian optimization.
    - Stores the collected data (inputs, tasks, targets) and the best-found points for the trial.
- After all trials are complete, it saves the aggregated results (`data_sets` and `bests`) to a pickle file.
"""


import torch
from utils.utils import unnormalize
import utils.utils
from utils.utils import standardize
from utils.utils import sample_from_task, concat_data
import utils.get_robust_gp
from bo.bo_loop import BayesianOptimization
from utils.functions import MixedMTRKHSFunction
import random
import numpy as np
import pickle
import argparse
from gpytorch.kernels import RBFKernel


function_name = "ICM"
d = 4
data_sets = []
bests = []
torch.set_default_dtype(torch.float64)
seeds = [73, 1, 92, 23, 45, 67, 89, 12, 34, 56, 78, 90, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888]
for seed in seeds:
    
    torch.manual_seed(seed)
    random.seed = seed
    np.random.seed(seed)
    delta_max = .05
    rho_max = .05
    tau = 0.001
    num_tsks = 2

    norm_bounds = torch.tensor([[-1.0], [1.0]]).repeat(1,d)
    kernel = RBFKernel(ard_num_dims=d)
    kernel.lengthscale = torch.ones(d)*0.2
    obj = MixedMTRKHSFunction(cor=[.85,.0], B=[30.5,30.5/3],id_norm=True, only_task_2=False, bounds=norm_bounds, kernel=kernel)     
    bounds = obj.bounds
    obj.plot()
    T = -1
    tasks = [0,1]

    nruns = 100 # number of optimization runs
    norm_x0=torch.tensor(torch.load(f'data/x_init_dim{d}/X_init_MTRKHSFunction_dim{d}_{seed}.pt')['X_init']).view(1,d)

    tasks = [0]
    num_sup_task_samples = 1
    num_acq_samps = [1]

    # evalaute initial point for all tasks
    train_targets = torch.zeros(num_tsks, 1)
    for j in range(num_tsks):
        train_targets[j, ...] = obj.f(norm_x0, j)
    train_tasks = torch.arange(num_tsks).unsqueeze(-1)
    norm_train_inputs = norm_x0.repeat(num_tsks, 1)

    # evalaute supplementary tasks
    for k in range(1, num_tsks):
        x, t, y = sample_from_task(
            obj,
            [k],
            norm_bounds + torch.tensor([[obj.max_disturbance], [-obj.max_disturbance]]),
            n=2 * num_sup_task_samples,
        )
        norm_train_inputs, train_tasks, train_targets = concat_data(
            (x, t, y), (norm_train_inputs, train_tasks, train_targets)
        )
    bo = BayesianOptimization(
        obj, tasks, norm_bounds, T, num_acq_samps
    )
    T_stdizd = bo.norm_threshold
    norm_train_targets = standardize(train_targets, train_task=train_tasks, threshold=T/T_stdizd)
    gp = utils.utils.build_mtgp(
        (norm_train_inputs, train_tasks),
        norm_train_targets, model_typ=function_name)
    covar = torch.zeros(nruns, num_tsks, num_tsks)
    beta_ = torch.zeros(nruns)
    mod_runs = 2
    for run in range(nruns):
        sqrtbeta = torch.sqrt(utils.get_robust_gp.beta_bayes(norm_bounds,tau,delta_max))
        robust_gp = gp
        robust_gp.task_covar_module._set_covar_factor(torch.eye(num_tsks))
        bo.update_gp(robust_gp, sqrtbeta)
        norm_train_inputs, train_tasks, norm_train_targets = bo.step()
        print(f"Threshold: {T_stdizd}")
        print(f"Min {robust_gp.train_targets[robust_gp.train_inputs[0][:,-1]==0].min()}")

        gp = utils.utils.build_mtgp(
            (norm_train_inputs, train_tasks), norm_train_targets, model_typ=function_name)
        
    # add data
    train_inputs = unnormalize(norm_train_inputs, bounds)
    train_targets = bo.unstd_train_targets
    print(
        f"Best value: {round(bo.best_y[-1],3)} at input: {unnormalize(bo.best_x[-1],bounds).round(decimals=3)}"
    )
    data_sets.append([train_inputs,train_tasks,train_targets])
    bests.append([bo.best_x,bo.best_y])

file = open(f"data/RKHS_ST/Single_Task_dim{d}_3.obj", "wb")
sets = {"data_sets": data_sets, "bests": bests}
pickle.dump(sets, file)
file.close()
