#!/usr/bin/env python3

"""
This module contains various classes for defining and working with different types of functions, including RKHS functions, 
multi-task RKHS functions, and several synthetic benchmark functions for optimization.
Classes:
    RKHSFunction:
        Represents a Reproducing Kernel Hilbert Space (RKHS) function.
        Methods:
            __init__(self, bounds, B, n, ns, kernel):
                Initializes the RKHS function with given parameters.
            f(self, x):
                Evaluates the RKHS function at given points.
            plot(self):
                Plots the RKHS function over a grid of points.
    MTRKHSFunction:
        Represents a multi-task RKHS function.
        Methods:
            __init__(self, bounds, B, n, ns, kernel, num_tsks):
                Initializes the multi-task RKHS function with given parameters.
            cov(self, x1, x2, t1, t2):
                Computes the covariance between two sets of points and tasks.
            f(self, x, t):
                Evaluates the multi-task RKHS function at given points and tasks.
            plot(self):
                Plots the multi-task RKHS function over a grid of points.
    MTPowell:
        Represents a multi-task Powell function.
        Methods:
            __init__(self, dim, num_tsks, disturbance):
                Initializes the multi-task Powell function with given parameters.
            f(self, x, t):
                Evaluates the multi-task Powell function at given points and tasks.
    MTBranin:
        Represents a multi-task Branin function.
        Methods:
            __init__(self, num_tsks, disturbance):
                Initializes the multi-task Branin function with given parameters.
            f(self, x, t):
                Evaluates the multi-task Branin function at given points and tasks.
"""



import torch
from torch import Tensor
from gpytorch.kernels import MaternKernel, Kernel, RBFKernel
import matplotlib.pyplot as plt
from utils.utils import normalize, unnormalize
from abc import abstractmethod
from botorch.test_functions.synthetic import Branin, Powell
from botorch.utils.sampling import draw_sobol_samples

class RKHSAbstract:
    def __init__(self):
        self.ns = 2000
        self.dim = 1

    @abstractmethod
    def f(self, x):
        pass

    def plot(self):
        x_grid = torch.linspace(self.bounds[0,0], self.bounds[1,0], self.ns).view(-1, 1)
        y_grid = self.f(x_grid)
        plt.plot(x_grid.detach().numpy(), y_grid.detach().numpy())
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('Plot of f(x)')
        plt.show()
        return x_grid, y_grid

class MTRKHSAbstract:
    def __init__(self):
        self.ns = 2000
        self.dim = 1
        self.num_tsks = 2
        self.max_disturbance = 0.

    @abstractmethod
    def f(self, x):
        pass

    def plot(self):
        x_grid = torch.linspace(self.bounds[0,0], self.bounds[1,0], self.ns).view(-1, 1)
        t_grid = torch.hstack([torch.zeros(self.ns,1), torch.ones(self.ns,1)]).to(dtype=torch.int64)
        y_grid = torch.hstack([self.f(x_grid,t_grid[:,i]) for i in range(self.num_tsks)])
        fig, ax = plt.subplots()
        for i in range(self.num_tsks):
            ax.plot(x_grid.detach().numpy(), y_grid[:, i].detach().numpy(), label=f'f_{i}')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title('Plot of f(x)')
        ax.legend()
        plt.show()
        return x_grid, y_grid

class RKHSFunction(RKHSAbstract):
    def __init__(self, data = None, B=1, bounds = torch.tensor([[-1.],[1.]]), n = 16, kernel:Kernel = None):
        super(RKHSFunction, self).__init__()
        self.bounds = bounds
        self.dim = bounds.size(-1)
        if kernel == None:
            self.kernel = RBFKernel(ard_num_dims=self.dim)
            self.kernel._set_lengthscale(torch.tensor([[0.1]*self.dim]))
        else:
            self.kernel = kernel
        self.xt = draw_sobol_samples(bounds, 4**self.dim, 1).view(-1,self.dim)    
        if data is not None:
            self.xt, self.y = data
            K_tilde = self.kernel(self.xt, self.xt) + 1e-10*torch.eye(self.xt.size(0))
            self.alpha_tilde = torch.linalg.solve(K_tilde, self.y)
        else:
            self.alpha_tilde = torch.randn(4**self.dim, 1)
        self.alpha = self.alpha_tilde / torch.sqrt(self.alpha_tilde.T @ self.kernel(self.xt, self.xt).evaluate().to(dtype=torch.double) @ self.alpha_tilde) * B

    def f(self, x):
        return self.kernel(x, self.xt) @ self.alpha
    

class MixedRKHSFunction(RKHSAbstract):
    def __init__(self, B = None, num_mixtures = 2, data: tuple = None, bounds = torch.tensor([[-1.],[1.]]), n = 16):
        super(MixedRKHSFunction, self).__init__()
        if B == None:
            B = [1,0.1]
        self.bounds = bounds
        self.funs = [RKHSFunction(data = data, B = B[i], bounds = bounds, n = n) for i in range(num_mixtures)]
    
    def f(self, x):
        return torch.stack([fun.f(x) for fun in self.funs]).sum(dim=0)


class MTRKHSFunction(MTRKHSAbstract):
    def __init__(self, cor, only_task_2 = False, id_norm = False, data = None, B=1, bounds = torch.tensor([[-1.],[1.]]), n = 20, kernel:Kernel = None):
        super(MTRKHSFunction, self).__init__()
        self.dim = bounds.size(-1)
        if kernel == None:
            self.kernel = RBFKernel(ard_num_dims=self.dim)
            self.kernel._set_lengthscale(torch.tensor([[0.1]*self.dim]))
        else:
            self.kernel = kernel
        self.num_tsks = 2
        self.id_norm = id_norm
        self.index_kernel = torch.tensor([[1., cor], [cor, 1.]]) if not only_task_2 else torch.tensor([[1., 0.], [0., 0.]])
        self.bounds = bounds
        if id_norm:
            self.funcs = [RKHSFunction(data = data, B = B, bounds = bounds, n = n, kernel = self.kernel) for _ in range(self.num_tsks)] 
            self.bounds = bounds
            self.mix = torch.linalg.cholesky(self.index_kernel) if not only_task_2 else self.index_kernel
        else:
            self.xt = draw_sobol_samples(bounds, n*self.num_tsks, 1).view(-1,self.dim)
            self.tt = torch.hstack([torch.zeros(n), torch.ones(n)]).to(dtype=torch.int64).view(-1, 1)
            alpha_tilde = torch.randn(n * self.num_tsks, 1)
            self.alpha = alpha_tilde / torch.sqrt((alpha_tilde.T @ self.cov(self.xt, self.xt, self.tt,self.tt) @ alpha_tilde))*B


    def cov(self, x1, x2, t1, t2):
        return self.kernel(x1, x2) * self.index_kernel[t1, t2.T]


    def f(self, x, t):
        if not isinstance(t,Tensor):
            t = torch.tensor([t])
        if t.numel() != x.size(0):
            t = t.repeat(x.size(0))
        t = t.to(dtype=torch.int32).view(-1,1)
        if self.id_norm:
            vec = torch.hstack([self.funcs[i].f(x) for i in range(2)])
            res = self.mix @ vec.transpose(0,1)
            res = res[t.squeeze(),torch.arange(0,t.size(0))].view(-1,1)
        else:
            res = self.cov(x, self.xt, t, self.tt) @ self.alpha
        # print(f"Res size: {res.size()}")
        return res

class MixedMTRKHSFunction(MTRKHSAbstract):
    def __init__(self, cor = [.95,0.05], B = None, only_task_2 = False, dim = 1, id_norm = False, num_mixtures = 2, data: tuple = None, bounds = torch.tensor([[-1.],[1.]]), n = 20, kernel:Kernel = None):
        super(MixedMTRKHSFunction, self).__init__()
        if B == None:
            B = [1,0.3]
        self.bounds = bounds
        self.dim = bounds.size(-1)
        self.funs = [MTRKHSFunction(cor=cor[i], data = data,only_task_2=only_task_2 if i>0 else False, id_norm = id_norm, B = B[i], bounds = bounds, n = n, kernel=kernel) for i in range(num_mixtures)]
    
    def f(self, x, t):
        res = torch.stack([fun.f(x,t) for fun in self.funs])
        # torch.save(res, f"plot_scripts/data/res_t{t[0].item()}.pt")
        return res.sum(0)
    
    def plot(self):
        if self.dim ==1:
            super().plot()
        else:
            pass
    

    
class MTPowell:
    def __init__(self, dim = 4, num_tsks = 2, disturbance = 0.1) -> None:
        self.bounds = torch.tensor([[-4.], [5.]]).repeat(1,dim)
        self.dim = dim
        self.disturbance = torch.sign(torch.rand(1,self.dim)-.5)*disturbance
        self.obj = [Powell(dim, negate=True,bounds=self.bounds.T) for _ in range(num_tsks)]
        self.offset = torch.rand(num_tsks)*2*25.e3-25.e3
        self.offset[0] = 0.0
        self.max_disturbance = disturbance
    
    def f(self, x, t): 
        x = unnormalize(X = x if t == 0 else x-self.disturbance, bounds=self.bounds)
        return self.obj[t](X=x,noise=False).view(-1,1)
    

class MTBranin:
    def __init__(self,num_tsks = 2, disturbance = 0.1) -> None:
        self.bounds = torch.tensor([[-5.,0], [10,15]])
        self.dim = 2
        self.disturbance = torch.sign(torch.rand(1,self.dim)-.5)*disturbance
        self.obj = [Branin(self.dim, negate=True,bounds=self.bounds.T) for _ in range(num_tsks)]
        self.offset = torch.rand(num_tsks)*2*30-30
        self.offset[0] = 0.0
        self.max_disturbance = disturbance

    def f(self, x, t): 
        x = unnormalize(X = x if t == 0 else x-self.disturbance, bounds=self.bounds)
        return self.obj[t](X=x,noise=False).view(-1,1)  
    


