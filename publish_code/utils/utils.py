#/usr/bin/python3
""" This module provides a suite of utility functions for building, evaluating, and managing
multi-task and single-task Gaussian Process (GP) models using GPyTorch and BoTorch.
It includes functions for model construction with specific priors and kernel configurations,
data manipulation (standardization, normalization, concatenation), and visualization of
model posteriors in 1D and 2D.
Core Functionalities:
-   **Model Construction**: Functions to build specialized Multi-Task GP models (e.g., LMC)
    and standard Single-Task GP models with pre-configured hyperparameters.
-   **Data Handling**: Utilities for sampling initial data, concatenating datasets,
    separating data by task, and normalizing/standardizing data.
-   **Visualization**: Plotting functions to inspect the GP posterior mean and
    confidence intervals for 1D and 2D input spaces.
    build_mtgp(train_inputs, train_targets, mu, likelihood, model_typ):
        Constructs a multi-task Gaussian Process model based on the `MultiTaskGPLMC`
        architecture with specific kernel, mean, and prior configurations.
    build_stgp(train_inputs, train_targets, likelihood):
        Constructs a single-task Gaussian Process model using `botorch.models.MultiTaskGP`
        configured for a single task.
    plot_post(gp, task, test_x, sqrtbeta, threshold):
        Plots the 1D posterior distribution (mean and confidence interval) for a
        specified task from a trained GP model.
    plot_post2D(gp, task, bounds, samps, sqrtbeta):
        Generates a 3D wireframe plot of the 2D posterior distribution for a
        specified task from a trained GP model.
    _mesh_helper(bounds, samps):
        A private helper function to create a 2D mesh grid from boundary definitions.
    sample_from_task(obj, tasks, bounds, n, data):
        Draws initial samples from an objective function for one or more tasks using
        Sobol sampling.
    concat_data(data, mem):
        Concatenates a new tuple of data tensors with an existing one.
    standardize(train_y, train_task, threshold):
        Scales target values by dividing them by a given threshold. Can operate on
        data from multiple tasks.
    unstandardize(norm_train_y, train_task, threshold):
        Reverses the standardization process by multiplying target values by the
        given threshold.
        Splits a combined dataset into separate lists of tensors, one for each task.
        Sorts training data points.
    normalize(x, bounds):
        Normalizes input data `x` from its original `bounds` to the range [-1, 1].
    unnormalize(x, bounds):
        Unnormalizes input data `x` from the range [-1, 1] back to its original `bounds`.
        """



import torch
from torch import Tensor
from botorch.utils import draw_sobol_samples
from botorch.models import MultiTaskGP
from cov.kernels import expDot
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from matplotlib import pyplot as plt
from model.model import MultiTaskGPLMC
from cov.task_cov import IndexKernelAllPriors
from gpytorch.means import MultitaskMean
from gpytorch.means import ConstantMean
from utils.priors import LKJCholeskyFactorPriorPyro
from copy import deepcopy
from gpytorch.constraints import GreaterThan, Interval

    
def build_mtgp(train_inputs, train_targets, mu=None, likelihood = GaussianLikelihood(), model_typ = None):
    num_tasks = train_inputs[1].max().to(dtype=torch.int32).item()+1
    d = train_inputs[0].shape[-1]
    train_inputs = torch.hstack(train_inputs).squeeze() 
    gp = MultiTaskGPLMC(
        train_inputs,
        train_targets,
        task_feature=-1,
        likelihood=likelihood,
        covar_module=ScaleKernel(
            MaternKernel(nu=2.5,ard_num_dims=d)),
        task_covar_module=IndexKernelAllPriors(
            num_tasks, num_tasks,
            covar_factor_prior=LKJCholeskyFactorPriorPyro(num_tasks,0.1)),
        mean_module=MultitaskMean([ConstantMean()],num_tasks)
    )
    chol_covar = torch.linalg.cholesky(torch.tensor([[1.,.99],[.99,1.]]))
    gp.task_covar_module._set_covar_factor(chol_covar)
    gp.task_covar_module.covar_factor.requires_grad = False
    gp.second_task_covar_module.covar_factor.requires_grad = False
    gp.mean_module.base_means[0].constant = 0.
    gp.mean_module.base_means[1].constant = 0.

    if "LMC" in model_typ:
        gp.second_covar_module._set_outputscale(1e-1) # sigma_f^2
        gp.covar_module.outputscale = torch.tensor([1.])

    else:
        gp.second_covar_module._set_outputscale(1e-8) # sigma_f^2
        gp.second_covar_module.raw_outputscale.requires_grad = False
        gp.covar_module.outputscale = torch.tensor([1.1])

    gp.likelihood.noise = 1e-4
    gp.covar_module.base_kernel.lengthscale = torch.tensor([.1]*d)    
    return gp

def build_stgp(train_inputs, train_targets, likelihood = GaussianLikelihood()):
    d = train_inputs[0].shape[-1]
    gp = MultiTaskGP(
        torch.hstack(train_inputs),
        train_targets,
        task_feature=-1,
        likelihood=likelihood,
        covar_module=ScaleKernel(MaternKernel(nu=2.5,ard_num_dims=train_inputs[0].shape[-1])),
        mean_module=ConstantMean()
    )
    gp.task_covar_module.covar_factor.data = torch.tensor([[1.]])
    gp.task_covar_module.covar_factor.requires_grad = False
    gp.mean_module.constant = -0.5
    gp.mean_module.constant.requires_grad = False
    gp.likelihood.noise = 1e-4
    gp.covar_module.base_kernel.lengthscale = torch.tensor([.08]*d)
    gp.covar_module.outputscale = torch.tensor([.5])
    gp.covar_module.register_constraint("raw_outputscale", Interval(0.1,1.))
    return gp

def plot_post(gp, task, test_x, sqrtbeta=None, threshold=None):
    figure = plt.figure(1)
    figure.clf()
    with torch.no_grad():
        posterior = gp.posterior(test_x.reshape( -1, 1, 1), output_indices=task)
        ymean = posterior.mean.squeeze()
        if sqrtbeta != None:
            std_dev = posterior.variance.squeeze().sqrt()
            u = ymean + sqrtbeta * std_dev
            l = ymean - sqrtbeta * std_dev
        else:
            l, u = posterior.mvn.confidence_region()
            u = u.detach().numpy(); l = l.detach().numpy()
    ax = figure.add_subplot(1, 1, 1)
    x = gp.train_inputs[0]
    i = x[:, -1]
    x = x[:, 0:-1]
    i = i.squeeze()
    y = gp.train_targets
    ax.fill_between(test_x, u.squeeze(), l.squeeze(), alpha=0.2, color="C0")
    ax.plot(test_x, ymean, "C0")
    ax.plot(x[i == 1], y[i == 1], "xC1")
    x1 = x[i == 0]; y1 = y[i == 0]
    ax.plot(x1[:-1], y1[:-1], "xC2")
    ax.plot(x1[-1],y1[-1],'oC2')
    if threshold is not None:
        ax.plot(test_x,threshold*torch.ones_like(test_x),'--')
    plt.show()


def plot_post2D(gp,task,bounds,samps=101,sqrtbeta=None):
    figure = plt.figure(1)
    figure.clf()
    sqrtbeta = 2 if sqrtbeta == None else sqrtbeta
    X, Y = _mesh_helper(bounds,samps)
    test_x = torch.cat((X.reshape(X.numel(),1), Y.reshape(Y.numel(),1)),-1)
    with torch.no_grad():
        posterior = gp.posterior(test_x, output_indices=task)
        ymean, yvar = posterior.mean.squeeze(-1), posterior.variance.squeeze(-1)
    ax = figure.add_subplot(1,1,1,projection='3d')
    ymean = ymean.reshape(X.size())
    yvar = yvar.reshape(X.size())
    ax.plot_wireframe(X.cpu().detach().numpy(),Y.cpu().detach().numpy(),(ymean+sqrtbeta*torch.sqrt(yvar)).cpu().detach().numpy(), color='r', rstride=10, cstride=10)
    ax.plot_wireframe(X.cpu().detach().numpy(),Y.cpu().detach().numpy(),(ymean-sqrtbeta*torch.sqrt(yvar)).cpu().detach().numpy(), color='b', rstride=10, cstride=10)
    x = gp.train_inputs[0]
    i = x[:,-1]
    x = x[:,0:-1]
    i = i.squeeze()
    y = gp.train_targets
    x1 = x[i==0].cpu().detach().numpy()
    x2 = x[i==1].cpu().detach().numpy()
    ax.scatter(x1[:,0], x1[:,1] ,y[i==0].cpu().detach().numpy(),"+r")
    ax.scatter(x2[:,0], x2[:,1] ,y[i==1].cpu().detach().numpy(),"+b")
    plt.show()


def _mesh_helper(bounds,samps=101):
    x1 = torch.linspace(bounds[0,0],bounds[1,0],samps)
    x2 = torch.linspace(bounds[0,1],bounds[1,1],samps)
    return torch.meshgrid(x1, x2, indexing='ij')


def sample_from_task(obj, tasks, bounds, n=5, data = None):
    with torch.no_grad():
        for i in tasks:
            ni = n[i] if isinstance(n,list) else n
            X_init = draw_sobol_samples(bounds = bounds, n=ni, q=1).reshape(ni,bounds.size(-1))
            data = concat_data((X_init,
                        i*torch.ones(X_init.size(0),1),
                        obj.f(X_init,i)),data)
    return data


def concat_data(data, mem = None):
    with torch.no_grad():
        if mem is None:
            return data
        else:
            return tuple([torch.vstack((mem,data)) for mem,data in zip(mem,data)])
    
def standardize(train_y, train_task = None, threshold = 1.):
    threshold = abs(threshold)
    if train_task != None:
        train_task = train_task.view(-1).to(dtype=torch.int32)  
        num_tasks = max(train_task)+1
        norm_train_y = deepcopy(train_y)
        for i in range(num_tasks):
            norm_train_y[train_task==i] = train_y[train_task==i]/threshold
    else:
        norm_train_y = train_y/threshold
    return norm_train_y

def unstandardize(norm_train_y, train_task = None, threshold = 1.):
    threshold = abs(threshold)
    if train_task != None:
        train_task = train_task.view(-1).to(dtype=torch.int32)
        num_tasks = max(train_task)+1
        train_y = deepcopy(norm_train_y)
        for i in range(num_tasks):
            train_y[train_task==i] = norm_train_y[train_task==i]*threshold
    else:
        train_y = norm_train_y*threshold
    return train_y

# # Standardize and unstandardize data which respect to primary task
# def standardize(train_y, mu = None, std = None, train_task = None):
#     if isinstance(train_y,Tensor):
#         train_y = deepcopy(train_y.detach())
#     if mu == None or std == None:
#         if train_task != None:
#             train_task = train_task.squeeze().to(dtype=torch.int32)
#             num_tasks = max(train_task)+1
#             mu = torch.hstack([torch.mean(train_y[train_task==i]) for i in range(num_tasks)])
#             norm_train_y = torch.cat([train_y[train_task==i]-mu[i] for i in range(num_tasks)])
#             std = torch.std(norm_train_y).nan_to_num(1.0).view(-1,1)
#             train_y = (train_y-mu[0])/std[0]
#             print(f"Std: {std}"); print(f"Mean: {mu}")
#             return train_y, mu, std 
#         else:
#             std,mu = torch.std_mean(train_y)
#             print(f"Std: {std.item()}")
#             print(f"Mean: {mu.item()}")
#             return (train_y-mu)/std,mu.view(-1,1),std.view(-1,1)
#     norm_train_y = (train_y-mu[0])/std[0]
#     return norm_train_y, mu, std

# def unstandardize(train_y, mu, std, train_task = None):
#     if train_task != None and mu.numel() > 1:
#         train_task=train_task.squeeze().to(dtype=torch.int32)
#         train_y = deepcopy(train_y)
#         num_tasks = train_task.max()+1
#         for i in range(num_tasks):
#             train_y[train_task==i] = train_y[train_task==i]*std[0] + mu[0]
#         return train_y
#     else:
#         train_y = train_y*std[0] + mu[0]
#         return train_y


def seperate_data(train_inputs,train_targets):
    train_x, train_t = train_inputs[:,:-1], train_inputs[:,-1:]
    num_tasks = train_t.max().to(dtype=torch.int32).item()+1
    d = train_x.shape[-1]
    sep_train_x = [train_x[train_t==i].view(-1,d) for i in range(num_tasks)]
    sep_train_y = [train_targets[train_t.squeeze()==i].view(-1,1) for i in range(num_tasks)]
    return sep_train_x,sep_train_y


def sort_data(train_inputs,train_targets):
    for i in range(len(train_inputs)):
        train_inputs[i],indices = train_inputs[i].sort()
        train_inputs[i] = train_inputs[i][indices]
        train_targets[i] = train_targets[i][indices]
    return train_inputs,train_targets

# noramlize data to [-1,1]^d
def normalize(x, bounds):
    return 2*(x-bounds[0])/(bounds[1]-bounds[0])-1

def unnormalize(x, bounds):
    return 0.5*(bounds[1]-bounds[0])*x+0.5*(bounds[1]+bounds[0])
    
