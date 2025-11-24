from gpytorch.kernels import Kernel
import torch

class expDot(Kernel):
    is_stationary = False
    has_lengthscale = True
    def __init__(self, ard_num_dims, batch_shape = None, lengthscale_prior = None, lengthscale_constraint = None, eps = 0.000001, **kwargs):
        super().__init__(ard_num_dims, batch_shape, None, lengthscale_prior, lengthscale_constraint, eps, **kwargs)


    def forward(self, x1, x2, diag = False, last_dim_is_batch = False, **params):
        if last_dim_is_batch:
            raise NotImplementedError
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        covar = x1_.matmul(x2_.transpose(-1,-2))
        covar.exp_()
        if diag:
            return covar.diagonal(dim1 = -2, dim2 = -1)
        return covar

    