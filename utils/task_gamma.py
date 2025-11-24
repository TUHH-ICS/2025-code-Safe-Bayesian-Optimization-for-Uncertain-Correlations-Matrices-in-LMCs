#!/usr/bin/env python3

"""This module provides utility functions for working with task covariance matrices
in the context of multi-task Bayesian optimization using Gaussian Processes.
The functions are designed to analyze and select from a distribution of task
covariance matrices sampled from a model's posterior.
    get_barbeta(sampmods, maxsqrtbeta, rho_max):
        Computes lambda and gamma values to find a robust task covariance matrix.
        It identifies a covariance matrix from a set of samples that is robust
        against uncertainty, considering a trade-off parameter `maxsqrtbeta` and
        a quantile `rho_max`.
    get_covar_factors(sampmods):
        Computes the Cholesky decomposition (lower triangular) of the task
        covariance matrices from the sampled models.
    get_gamma(sampmods):
        Computes a matrix of norms representing the dissimilarity between pairs
        of sampled task covariance matrices. This is used to quantify the
        difference in the underlying task correlation structures.
        Selects a subset of candidate covariance matrices from the sampled models
        by sorting them based on their determinant and taking a fixed proportion.
    get_nu(sampmods):
        Computes the square root of the mean squared error between posterior means
        calculated with different sampled task covariance matrices. This quantifies
        the impact of covariance uncertainty on the posterior mean prediction.
    norm_scales(sampmods):
        Computes a matrix of ratios between the output scales of the sampled models.
    post_mean_SE(sampmods):
        Computes the squared error between the posterior means of different
        sampled models, normalized by the likelihood noise.
        Generates and displays a histogram of the determinants of the task
        covariance matrices from the sampled models. This helps visualize the
        distribution of the sampled covariance volumes.
"""


import torch
from math import floor
from matplotlib import pyplot as plt


def get_barbeta(sampmods, maxsqrtbeta, rho_max: float):
    covar_factors = sampmods.task_covar_module.covar_factor
    samps = sampmods.task_covar_module.covar_factor.size(0)
    plot_sampmods_det(sampmods)
    indmax = floor((1-rho_max)*samps)
    with torch.no_grad():
        gamma = get_gamma(sampmods)
        nu = get_nu(sampmods)
    total = nu + gamma * maxsqrtbeta
    total,inds = total.sort(dim=-1)
    Id = total[:,indmax].argmin()
    thprime = covar_factors[Id]
    gamma = gamma[Id,inds[Id,indmax]].detach()
    nu = nu[Id,inds[Id,indmax]].detach()
    return gamma, nu, thprime, total[Id,indmax]

def get_covar_factors(sampmods):
    covar = sampmods.task_covar_module._eval_covar_matrix()
    return torch.linalg.cholesky(covar, upper=False).squeeze()

def get_gamma(sampmods):
    covar_factors = get_covar_factors(sampmods)    
    covs = torch.linalg.solve(covar_factors.unsqueeze(1), covar_factors.unsqueeze(0))
    norms = torch.linalg.norm(covs,2,dim=(-2,-1))
    lower_right = covar_factors.unsqueeze(1)-covar_factors.unsqueeze(0)
    
    norms = torch.where(lower_right[:,:,-1,-1]>=0, torch.tensor(1.0), norms)
    return norms

def get_covar_candidates(sampmods):
    covar = sampmods.task_covar_module._eval_covar_matrix()
    num_samps = covar.size(0)
    _,inds = torch.linalg.det(covar).sort(descending=False)
    candiate_dict = {"covar":covar[inds[:round(1.*num_samps)]], "inds":inds[:round(1.*num_samps)]}
    return candiate_dict

def get_nu(sampmods):
    sampmods.train()
    covar_factors = sampmods.task_covar_module._eval_covar_matrix().detach()
    L = covar_factors.unsqueeze(1).matmul(torch.linalg.solve(covar_factors.unsqueeze(1), covar_factors.unsqueeze(0)))
    train_inputs = sampmods.train_inputs[0]
    train_targets = sampmods.train_targets
    train_tasks = train_inputs[0,:,-1].to(dtype=torch.int32)
    noise = sampmods.likelihood.noise.detach()
    K = sampmods(train_inputs).covariance_matrix
    Kd = K + noise * torch.eye(K.size(-1)).unsqueeze(0)
    alpha = torch.linalg.solve(Kd, train_targets.unsqueeze(-1))
    k1 = sampmods.covar_module(train_inputs[:,:,:-1]).evaluate() # Gram matrix with B^(i) = eye(u)
    task_covs = L[...,train_tasks,train_tasks.unsqueeze(-1)]
    K2 = k1.mul(task_covs)
    term1 = alpha.transpose(-1,-2).matmul(K).matmul(alpha)
    term2 = alpha.unsqueeze(1).transpose(-1,-2).matmul(K).matmul(alpha)
    term3 = alpha.transpose(-1,-2).matmul(K2).matmul(alpha)
    norm_diff = (term1.unsqueeze(1) -2*term2 + term3).squeeze()
    mean_se = post_mean_SE(sampmods)
    return norm_diff.add(mean_se).maximum(torch.tensor(1e-12)).sqrt()


def norm_scales(sampmods):
    out_scales = sampmods.second_covar_module.outputscale.detach().view(-1,1)
    mat = out_scales / out_scales.T
    return mat

def post_mean_SE(sampmods):
    noise =sampmods.likelihood.noise.detach()
    (train_inputs,) = sampmods.train_inputs
    sampmods.eval()
    mu1 = sampmods(train_inputs).mean
    mu0 = mu1
    normdiff = torch.norm(mu0.unsqueeze(1) - mu1.unsqueeze(0), 2,dim=-1) ** 2 / noise
    return normdiff


def plot_sampmods_det(sampmods):
    covar = sampmods.task_covar_module._eval_covar_matrix()
    dets = torch.linalg.det(covar)
    _, ax = plt.subplots()
    ax.hist(dets.detach().numpy())
    ax.set_title("Determinants of Covariance Matrices")
    ax.set_xlabel("Determinant")
    ax.set_ylabel("Frequency")
    plt.show()