#!/usr/bin/env python3

"""
This module provides functions for robust Gaussian Process (GP) modeling.

It includes methods for constructing robust GP models using Bayesian principles and for
calculating parameters like beta and lambda for both Bayesian and frequentist
approaches.

Functions:
    bayesian_robust_gp: Constructs a robust GP model using Bayesian methods.
    beta_bayes: Computes the beta parameter for a Bayesian robust GP.
"""



import torch
from utils.task_gamma import get_barbeta
from copy import deepcopy
from torch import Tensor


def bayesian_robust_gp(
    sampmods,
    model0,
    bounds,
    delta_max: float = 0.05,
    tau: float = 0.01,
    rho_max = None):

    if rho_max is None:
        rho_max = delta_max
    maxsqrtbeta = beta_bayes(bounds, tau, delta_max=delta_max).sqrt()
    gamma, lambda_, sigprime, total = get_barbeta(sampmods, maxsqrtbeta, rho_max=rho_max
    )
    sigmaf = model0.covar_module.outputscale.detach()
    print(f"Sigma_f: {sigmaf}")
    print(f"Correlation Matrix: {sigprime@sigprime.T}")
    print(f"Uncertainty at supplementary task: {total*sigmaf}")
    model0.task_covar_module._set_covar_factor(sigprime)
    robustmodel = deepcopy(model0)
    print(f"lambda:{lambda_}")
    print(f"gamma: {gamma}")
    sqrtbeta = 1 * maxsqrtbeta + lambda_
    print(f"sqrtbeta: {sqrtbeta}")
    return robustmodel, sqrtbeta


def beta_bayes(
    bounds: Tensor = torch.arange(2).unsqueeze(-1),
    tau: float = 0.01,
    delta_max: float = 0.05,
):
    M = torch.hstack([torch.ceil((bu - bl) / (2*tau) + 1) for bl, bu in bounds.T])
    m = M.prod()
    beta = 2 * torch.log(m / delta_max)
    print(f"beta: {beta}")
    return beta


