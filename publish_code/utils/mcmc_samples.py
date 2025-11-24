#!/usr/bin/env python3

"""
This module provides a utility function for obtaining MCMC samples from a Gaussian Process model using Pyro.

Functions:
    run_mcmc(gp, num_samples=100, warmup_steps=100):
        Runs MCMC sampling on the given Gaussian Process model and returns the samples and diagnostics.

        Parameters:
            gp (gpytorch.models.ExactGP): The Gaussian Process model to sample from.
            num_samples (int, optional): The number of MCMC samples to generate. Default is 100.
            warmup_steps (int, optional): The number of warmup steps for the MCMC sampler. Default is 100.

        Returns:
            tuple: A tuple containing:
                - samp_temp (dict): A dictionary of MCMC samples.
                - diagnostics (dict): A dictionary of diagnostics information from the MCMC run.
"""

import gpytorch
import pyro
from copy import deepcopy
from pyro.infer.mcmc import NUTS, MCMC
from torch import hstack, all, eye, vstack
from utils.optim import optimize_gp


def run_mcmc(gp,num_samples=100, warmup_steps=100):
    train_inputs = gp.train_inputs[0]
    train_targets = gp.train_targets
    gppyro = deepcopy(gp)
    gppyro.task_covar_module.add_prior()
    # gppyro.second_task_covar_module.add_prior()
    gppyro.train()

    def pyro_model(x, y):
        with gpytorch.settings.fast_computations(False, False, False):
            sampled = gppyro.pyro_sample_from_prior()
            output = sampled.likelihood(sampled(x))
            pyro.sample("obs", output, obs=y.squeeze())
        return y

    nuts_kernel = NUTS(pyro_model,step_size=.7,target_accept_prob=.9, max_tree_depth=3, adapt_step_size=True)
    mcmc_run = MCMC(
        nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps, disable_progbar=False
    )
    mcmc_run.run(train_inputs, train_targets)
    diagnostics = mcmc_run.diagnostics()
    samp_temp = mcmc_run.get_samples()
    inds = hstack([all((samp_temp['task_covar_module.covar_factor_prior'][i]@samp_temp['task_covar_module.covar_factor_prior'][i].T)[0]>=0) for i in range(num_samples)])
    for key in samp_temp.keys():
        samp_temp[key] = samp_temp[key][inds]

    # gppyro.pyro_load_from_samples(samp_temp)
    return samp_temp, diagnostics

def get_samples(gp, min_samples = 50, num_samples=100, warmup_steps=100):
    counter = 0
    num_tasks = gp.task_covar_module.covar_factor.size(-1)
    samp_gp = deepcopy(gp) 
    # samp_gp.likelihood.noise = 1e-4
    samp_gp,_,_ = optimize_gp(
                samp_gp, mode=1, max_iter=200) 
    samples, diagnostics = run_mcmc(
        gp=samp_gp, num_samples=num_samples, warmup_steps=warmup_steps
    )

    #update the two learnable hyperparameters
    # gp.mean_module.base_means[1].constant = samp_gp.mean_module.base_means[1].constant.detach()
    # gp.second_covar_module.outputscale = samp_gp.second_covar_module.outputscale.detach()

    #start sampling
    while samples["task_covar_module.covar_factor_prior"].shape[0] <= min_samples:
        if counter > 10:
            samples["task_covar_module.covar_factor_prior"] = (
                eye(num_tasks).unsqueeze(0).repeat(2, 1, 1)
            )
            break
        samples0, diagnostics = run_mcmc(
            gp=samp_gp, num_samples=50, warmup_steps=100
        )
        samples = {
            key: vstack((samples[key], samples0[key]))
            for key in samples.keys()
        }
        counter += 1
    return samples, diagnostics
