#!/usr/bin/env python3

"""
This script runs a Multi-Task Bayesian Optimization (MTBO) experiment.

The script iterates through a set of random seeds to ensure reproducibility. For each seed, it performs the following steps:
1.  Initializes the optimization environment, including the multi-task objective function (MixedMTRKHSFunction), optimization bounds, and various parameters for robust optimization.
2.  Generates an initial dataset by evaluating a starting point across all tasks and sampling additional points from supplementary tasks.
3.  Initializes a BayesianOptimization loop controller and a Multi-Task Gaussian Process (MTGP) model using the initial data.
4.  Enters the main optimization loop for a fixed number of iterations (`nruns`):
    a. Periodically, it updates the GP model to be robust against model uncertainty. This is done by generating MCMC samples from the current GP's posterior and using them to construct a robust GP.
    b. The BO controller uses the robust GP to select the next point to evaluate.
    c. The objective function is evaluated at the new point, and the data is used to update the base GP model for the next iteration.
5.  After the optimization loop finishes, it records the final results, including all evaluated points and the best-found solution.
6.  Finally, it saves the collected data and best results from all seed runs to a file using pickle for later analysis.
"""


import torch
from utils.utils import unnormalize
import utils.utils
from utils.utils import standardize
from utils.utils import sample_from_task, concat_data
from utils.mcmc_samples import get_samples
import utils.get_robust_gp
from bo.bo_loop import BayesianOptimization
from utils.functions import MixedMTRKHSFunction
from math import ceil
from copy import deepcopy
import random
import numpy as np
import pickle
import argparse
from gpytorch.kernels import RBFKernel

model_type = "LMC"  # or "ICM"
d = 4 # dimension
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
    tau = .001
    num_tsks = 2

    norm_bounds = torch.tensor([[-1.0], [1.0]]).repeat(1,d)
    kernel = RBFKernel(ard_num_dims=d)
    kernel.lengthscale = torch.ones(d)*0.2
    obj = MixedMTRKHSFunction(cor=[.85,.0], B=[30.5,10.5],id_norm=True, only_task_2=False, bounds=norm_bounds, kernel=kernel)     
    bounds = obj.bounds
    obj.plot()
    T = -1
    tasks = [0,1]

    nruns = 100 # number of optimization runs

    norm_x0=torch.tensor(torch.load(f'data/x_init_dim{d}/X_init_MTRKHSFunction_dim{d}_{seed}.pt')['X_init']).view(1,d)
    tasks = list(range(num_tsks))
    num_sup_task_samples = ceil(2 * d / (num_tsks - 1))
    num_acq_samps = [1]
    for _ in range(num_tsks - 1):
        num_acq_samps.append(num_sup_task_samples)

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
        norm_train_targets, model_typ=model_type)
    covar = torch.zeros(nruns, num_tsks, num_tsks)
    beta_ = torch.zeros(nruns)
    mod_runs = 2
    for run in range(nruns):
        if run >= 45:
            bo.num_acq_samps = [1] * num_tsks
            mod_runs = 5
        # gp,_,_ = optimize_gp(
        #             gp, mode=1, max_iter=200)  # get MAP estimate
        if run ==0:
            sqrtbeta = torch.sqrt(utils.get_robust_gp.beta_bayes(norm_bounds,tau,delta_max))
            robust_gp = gp
            robust_gp.task_covar_module._set_covar_factor(torch.eye(num_tsks))
        elif (run <= 10 and run % mod_runs == 0) or (run % mod_runs == 0):

            samples,_ = get_samples(gp, min_samples=50, num_samples=100, warmup_steps=10)
            sample_models = deepcopy(gp)
            sample_models.task_covar_module.add_prior()
            sample_models.pyro_load_from_samples(samples)

            robust_gp, sqrtbeta = utils.get_robust_gp.bayesian_robust_gp(
                sample_models, gp, norm_bounds, delta_max=delta_max, tau=tau, rho_max=rho_max
            )
            noise = robust_gp.likelihood.noise.detach()
            varf = robust_gp.covar_module.outputscale.detach()
            del sample_models
        else:
            chol_covar = robust_gp.task_covar_module.covar_factor.detach()
            robust_gp = gp
            robust_gp.task_covar_module._set_covar_factor(chol_covar)
        covar[run, ...] = robust_gp.task_covar_module._eval_covar_matrix()
        beta_[run] = sqrtbeta
        print([robust_gp.mean_module.base_means[i].constant for i in range(num_tsks)])
        bo.update_gp(robust_gp, sqrtbeta)
        norm_train_inputs, train_tasks, norm_train_targets = bo.step()
        print(f"Threshold: {T_stdizd}")
        print(f"Min {robust_gp.train_targets[robust_gp.train_inputs[0][:,-1]==0].min()}")

        gp = utils.utils.build_mtgp(
            (norm_train_inputs, train_tasks), norm_train_targets, model_typ=model_type)
    # add data
    train_inputs = unnormalize(norm_train_inputs, bounds)
    train_targets = bo.unstd_train_targets
    print(
        f"Best value: {round(bo.best_y[-1],3)} at input: {unnormalize(bo.best_x[-1],bounds).round(decimals=3)}"
    )
    data_sets.append([train_inputs,train_tasks,train_targets])
    bests.append([bo.best_x,bo.best_y])

file = open(f"data/RKHS/{model_type}_dim{d}_3.obj", "wb")
sets = {"data_sets": data_sets, "bests": bests}
pickle.dump(sets, file)
file.close()
