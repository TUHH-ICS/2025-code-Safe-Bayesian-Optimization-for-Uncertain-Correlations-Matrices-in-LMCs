#!/usr/bin/env python3
"""A Multi-Task Gaussian Process model using a sum of two Linear Models of Coregionalization (LMC).
This model extends the standard `botorch.models.MultiTaskGP` by employing a more
flexible covariance structure. The covariance between an output `i` at input `x`
and an output `j` at input `x'` is modeled as a sum of two coregionalized
kernels:
    K((x, i), (x', j)) = K_data1(x, x') * K_task1(i, j) + K_data2(x, x') * K_task2(i, j)
This structure allows the model to capture complex inter-task correlations by
combining two different covariance patterns. For example, one component can model a
primary shared structure across tasks, while the second component can model
task-specific variations or secondary effects.
The first component (K_data1, K_task1) is typically configured like a standard
Intrinsic Coregionalization Model (ICM). The second component (K_data2, K_task2)
provides additional flexibility. By default, the second task covariance kernel
is initialized as a fixed, non-trainable kernel to prevent identifiability issues,
but it can be customized for specific modeling needs.
This model is particularly useful in scenarios where a simple ICM is not
sufficient to describe the relationships between tasks.
"""




from typing import List
from botorch.models import MultiTaskGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.kernels import RBFKernel, ScaleKernel, IndexKernel
from gpytorch.module import Module
from gpytorch.priors.prior import Prior
from utils.priors import LKJCholeskyFactorPriorPyro
from gpytorch.priors import GammaPrior
from gpytorch.constraints import Interval
from gpytorch.distributions.multivariate_normal import MultivariateNormal
from torch import Tensor
from cov.task_cov import IndexKernelAllPriors
import torch
from math import sqrt


"""Multi-Task GP model with a sum of two LMC covariance structures.
    Args:
        train_X: A `n x d` or `b x n x d` tensor of training features. The last
            dimension `d` must include a task index column specified by `task_feature`.
        train_Y: A `n x m` or `b x n x m` tensor of training observations.
        task_feature: The index of the column in `train_X` that contains the task indices.
        train_Yvar: An optional `n x m` or `b x n x m` tensor of observed noise variances.
        mean_module: The mean function for the GP. Defaults to `ConstantMean`.
        covar_module: The data covariance kernel for the first LMC component (K_data1).
            Defaults to `ScaleKernel(RBFKernel)`.
        task_covar_module: The task covariance kernel for the first LMC component (K_task1).
            Defaults to `IndexKernel`.
        second_covar_module: The data covariance kernel for the second LMC component (K_data2).
            Defaults to `ScaleKernel(RBFKernel)`.
        likelihood: The likelihood for the model. Defaults to `GaussianLikelihood`.
        task_covar_prior: A prior on the first task covariance matrix `B_1`.
        output_tasks: A list of task indices to be modeled. If not provided, all tasks
            present in `train_X` are used.
        rank: The rank of the first task covariance matrix `B_1`.
        input_transform: An input transform to be applied to the training data.
        outcome_transform: An outcome transform to be applied to the training data.
    """
class MultiTaskGPLMC(MultiTaskGP):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        task_feature: int,
        train_Yvar: Tensor | None = None,
        mean_module: Module | None = None,
        covar_module: Module | None = None,
        task_covar_module: Module | None = None,
        second_covar_module: Module | None = None,
        likelihood: Likelihood | None = None,
        task_covar_prior: Prior | None = None,
        output_tasks: List[int] | None = None,
        rank: int | None = None,
        input_transform: InputTransform | None = None,
        outcome_transform: OutcomeTransform | None = None,
    ) -> None:
        super().__init__(
            train_X,
            train_Y,
            task_feature,
            train_Yvar,
            mean_module,
            covar_module,
            likelihood,
            task_covar_prior,
            output_tasks,
            rank,
            input_transform,
            outcome_transform,
        )
        if task_covar_module is not None:
            self.task_covar_module = task_covar_module
        if second_covar_module is not None:
            self.second_covar_module = second_covar_module
        else:
            self.second_covar_module = ScaleKernel(RBFKernel(ard_num_dims=train_X.size(-1)-1))
        
        if second_covar_module is not None:
            self.second_task_covar_module = second_covar_module
        else:
            num_task = self.task_covar_module.covar_factor.size(-1)
            self.second_task_covar_module = IndexKernelAllPriors(rank=num_task, num_tasks=num_task, const=0.0, covar_factor_prior=LKJCholeskyFactorPriorPyro(num_task, 100.0))
            cov = torch.zeros_like(self.task_covar_module._eval_covar_matrix())
            cov[...,0,0] = sqrt(1.05)
            self.second_task_covar_module._set_covar_factor(cov)
            self.second_task_covar_module._set_var(torch.zeros(num_task))
            self.second_task_covar_module.raw_var.requires_grad = False
            self.second_task_covar_module.covar_factor.requires_grad = False

    def forward(self,x: Tensor) -> MultivariateNormal:
        if self.training:
            x = self.transform_inputs(x)
        x_basic, task_idcs = self._split_inputs(x)
        # Compute base mean and covariance
        mean_x = self.mean_module(x_basic)
        mean_x = mean_x[...,torch.arange(task_idcs.size(-2)),task_idcs.reshape(-1,task_idcs.size(-2))[0]]
        covar_x = self.covar_module(x_basic)
        # Compute task covariances
        covar_i = self.task_covar_module(task_idcs)
        # Combine the two in an ICM fashion
        covar1 = covar_x.mul(covar_i)

        covar_i2 = self.second_task_covar_module(task_idcs)
        covar2 = covar_i2.mul(self.second_covar_module(x_basic))
        covar = covar1 + covar2

        return MultivariateNormal(mean_x, covar)
        
