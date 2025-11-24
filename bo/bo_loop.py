#!/usr/bin/env python3
"""
Provides a Bayesian Optimization (BO) class with multi-task and safety features.
The `BayesianOptimization` class handles the BO loop, data, acquisition,
and SafeOpt-based safety constraints.
"""

import torch
from torch import Tensor
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition import qUpperConfidenceBound
from utils.utils import concat_data, unstandardize, standardize
from botorch.acquisition.objective import ScalarizedPosteriorTransform

N_TOL = -1e-6

"""
    Manages a multi-task Bayesian Optimization loop with optional safety constraints.
    Suggests new points, updates the model, and tracks the best solutions.
    It can handle multiple tasks and enforce safety on a primary task.
    Attributes:
        obj: The objective function oracle.
        tasks (list): Task identifiers.
        bounds (Tensor): Search space bounds.
        threshold (float): Safety threshold for the primary task.
        num_acq_samps (list): Number of points to acquire per task.
        safeopt (bool): Enables SafeOpt-like strategy for 1D problems.
        run (int): Current BO loop iteration number.
        best_y (list): History of best objective values.
        best_x (list): History of best inputs.
        dim (int): Dimensionality of the input space.
        gp (gpytorch.models.GP): The Gaussian Process model.
        norm_threshold (float): Normalized safety threshold.
        train_inputs (Tensor): Training data inputs.
        train_tasks (Tensor): Task identifiers for training data.
        unstd_train_targets (Tensor): Unstandardized training targets.
        train_targets (Tensor): Standardized training targets.
        observed_max (list): Maximum observed value per task.
        sqrtbeta (Tensor): UCB/LCB exploration factor.
    """
class BayesianOptimization:

    """
        Initializes the BayesianOptimization instance.
        Args:
            obj: The objective function to be optimized.
            tasks (list): A list of task identifiers.
            bounds (Tensor): A (2, dim) tensor with search space bounds.
            threshold (float): The safety threshold for the objective function.
            num_acq_samps (list, optional): Samples per task. Defaults to [1, 1].
            safeopt (bool, optional): Use SafeOpt for 1D problems. Defaults to False.
        """
    def __init__(
        self,
        obj,
        tasks,
        bounds,
        threshold,
        num_acq_samps: list = [1, 1],
        safeopt=False
    ):
        self.obj = obj
        self.bounds = bounds
        self.threshold = threshold
        self.num_acq_samps = num_acq_samps
        self.tasks = tasks
        if len(self.num_acq_samps) != len(self.tasks):
            raise ValueError("Number of tasks and number of samples must match")
        self.run = 0
        self.best_y = []
        self.best_x = []
        self.dim = bounds.size(-1)
        self.gp = None
        self.norm_threshold = -1.0
        self.safeopt = safeopt

    """
        Performs one step of the Bayesian Optimization loop.
        This method finds the next point(s) to evaluate, queries the objective
        function, and updates the dataset.
        Returns:
            tuple[Tensor, Tensor, Tensor]: Updated training inputs, tasks, and targets.
        """

    def step(self):
        self.run += 1
        print("Run : ", self.run)
        print(f"Best value: {self.observed_max[0]: .3f}")
        print(f"Worst value: {self._get_min_observed()[0]: .3f}")
        if len(self.tasks) == 1:
            W = torch.eye(2)
        else:
            W = torch.eye(len(self.tasks))
        for i in self.tasks:
            if not self.safeopt:
                posterior_transform = ScalarizedPosteriorTransform(W[:, i].squeeze())
                new_point = self.get_next_point(i, posterior_transform)
            else:
                new_point = self.get_next_point(i, None)
            if i == 0:
                new_point_task0 = new_point
            if i != 0:
                new_point = torch.vstack((new_point, new_point_task0))
            new_result = self.obj.f(new_point, i)
            if i == 0:
                print(f"New Point: {new_point}")
                print(f"New Observation: {new_result}")
            self.train_inputs, self.train_tasks, self.unstd_train_targets = concat_data(
                (new_point, i * torch.ones(new_point.shape[0], 1), new_result),
                (self.train_inputs, self.train_tasks, self.unstd_train_targets),
            )
        self.train_targets = standardize(
            self.unstd_train_targets, train_task=self.train_tasks, threshold=self.threshold/self.norm_threshold
        )
        self.observed_max = self._get_max_observed()
        self.best_y.append(self.observed_max[0])
        self.best_x.append(self._get_best_input()[0])
        return self.train_inputs, self.train_tasks, self.train_targets
    
    """
        Defines the nonlinear inequality constraint for safe optimization.
        A point is safe if its LCB is above the normalized safety threshold.
        Args:
            input (Tensor): The input tensor to evaluate the constraint on.
        Returns:
            Tensor: A non-negative value indicates the constraint is satisfied.
        """

    def inequality_consts(self, input: Tensor):
        self.gp.eval()
        inputx = input.view(int(input.numel() / self.dim), self.dim)
        output = self.gp(torch.hstack((inputx, torch.zeros(inputx.size(0), 1))))
        val = (
            output.mean
            - output.covariance_matrix.diag().sqrt() * self.sqrtbeta
            - self.norm_threshold
        )
        return val.view(inputx.shape[0], 1)

    """
        Updates the GP model and related internal state.
        This is called after the GP model has been retrained with new data.
        Args:
            gp (gpytorch.models.GP): The new, trained Gaussian Process model.
            sqrtbeta (Tensor): The updated exploration-exploitation trade-off parameter.
        """
    def update_gp(self, gp, sqrtbeta):
        with torch.no_grad():
            self.train_inputs = gp.train_inputs[0][..., :-1]
            self.train_tasks = gp.train_inputs[0][..., -1:].to(dtype=torch.int32)
            self.train_targets = gp.train_targets.unsqueeze(-1)
            self.unstd_train_targets = unstandardize(
                self.train_targets, self.train_tasks, self.threshold/self.norm_threshold)
            self.sqrtbeta = sqrtbeta.detach()
        if self.gp is None:
            self.observed_max = self._get_max_observed()
            self.best_y.append(self.observed_max[0])
            self.best_x.append(self._get_best_input()[0])
        self.gp = gp
        pass
    """
        Performs a line search to find a feasible point from an initial condition.
        Searches along a random direction to find a better, safe starting point
        for acquisition function optimization.
        Args:
            initial_condition (Tensor): The starting points for the line search.
            maxiter (int, optional): Not used. Defaults to 20.
            step_size (float, optional): The range of the line search. Defaults to 2.0.
        Returns:
            Tensor: The updated, feasible initial conditions.
        """

    def _line_search(self, initial_condition, maxiter=20, step_size=2.0):
        k = 300
        direction = torch.randn(initial_condition.size())
        direction /= (
            torch.linalg.norm(direction, dim=-1, ord=2)
            .unsqueeze(-1)
            .repeat(1, 1, self.dim)
        )
        steps = torch.linspace(0, step_size, k).view(1, k, 1) - step_size / 2
        line_search = torch.clamp(initial_condition + steps * direction, self.bounds[0,:], self.bounds[1,:])
        inds = (self.inequality_consts(line_search) >= 0).view(initial_condition.size(0), -1)
        for id in range(inds.size(0)):
            true_indices = torch.nonzero(inds[id, :], as_tuple=False).squeeze()
            if true_indices.numel() == 0:
                continue
            mid_index = inds.size(1) // 2
            distances = torch.abs(true_indices - mid_index)
            max_distance_index = distances.argmax()
            farthest_true_index = true_indices[max_distance_index]
            if torch.all(inds[id, min(farthest_true_index, mid_index):max(farthest_true_index, mid_index) + 1]):
                initial_condition[id] = (
                    initial_condition[id]
                    + steps[:, farthest_true_index, :].squeeze() * direction[id]
                )
        print(initial_condition)
        return initial_condition
    """
        Retrieves the maximum observed objective value for each task.
        Returns:
            list[float]: A list of maximum observed values per task.
        """

    def _get_max_observed(self):
        return [
            torch.max(self.unstd_train_targets[self.train_tasks == i]).item()
            for i in self.tasks
        ]
    """
        Retrieves the minimum observed objective value for each task.
        Returns:
            list[float]: A list of minimum observed values per task.
        """
        
    def _get_min_observed(self):
        return [
            torch.min(self.unstd_train_targets[self.train_tasks == i]).item()
            for i in self.tasks
        ]
    """
        Retrieves the input corresponding to the best observed value for each task.
        Returns:
            list[Tensor]: A list of inputs that yielded the highest target per task.
        """
        
    def _get_best_input(self):
        return [
            self.train_inputs[self.train_tasks.squeeze() == i, ...][
                torch.argmax(self.train_targets[self.train_tasks == i])
            ]
            for i in self.tasks
        ]
    """
        Generates safe initial conditions for acquisition function optimization.
        Samples from previously observed points, weighted by their target values,
        and filters for safety.
        Returns:
            Tensor: A tensor of feasible initial conditions.
        """
        
    def _get_initial_cond(self):
        train_x0 = self.train_inputs[self.train_tasks.squeeze() == 0]
        train_x = self.train_inputs[self.train_tasks.squeeze() != 0]
        probabilities_task0 = torch.softmax(self.train_targets[self.train_tasks.squeeze() == 0].view(-1), dim=0)
        probabilities_task_other = torch.softmax(self.train_targets[self.train_tasks.squeeze() != 0].view(-1), dim=0)
        sampled_indices0 = torch.multinomial(probabilities_task0, num_samples=min(3, probabilities_task0.numel()), replacement=False)
        sampled_indices = torch.multinomial(probabilities_task_other, num_samples=2, replacement=False)
        sampled_train_inp = torch.vstack((train_x0[sampled_indices0], train_x[sampled_indices]))
        eqfull = self.inequality_consts(sampled_train_inp).squeeze()
        pot_cond = sampled_train_inp[eqfull >= 0, ...]
        return pot_cond.view(pot_cond.size(0), 1, self.dim)

    """
        Optimizes the acquisition function to find the next evaluation point.
        Uses constrained optimization for the primary task to ensure safety.
        Args:
            task (int): The task identifier.
            posterior_transform (ScalarizedPosteriorTransform or None): A transform
                applied to the posterior to scalarize multi-output models.
        Returns:
            Tensor: The candidate point(s) to evaluate next.
        """

    def get_next_point(self, task, posterior_transform):
        if task == 0:
            if self.bounds.size(-1) == 1:
                return self.safeopt(posterior_transform)
            else:
                init_cond = self._get_initial_cond()
                if init_cond.numel() == 0:
                    print(
                        "No feasible initial condition found. Randomly sampling a new one."
                    )
                    x_new = self.train_inputs[
                        self.train_targets[self.train_tasks == 0].argmax(), :
                    ].view(1, self.dim)
                    offset = torch.randn(1, self.dim) * 0.005
                    ind = (x_new + offset <= self.bounds[1, :].view(1, self.dim)) & (
                        x_new + offset >= self.bounds[0, :].view(1, self.dim)
                    )
                    x_new[ind] = x_new[ind] + offset[ind]
                    x_new[~ind] = x_new[~ind] - offset[~ind]
                    return x_new
                else:
                    try:
                        init_cond = self._line_search(init_cond)
                    except:
                        init_cond = init_cond
                acq = qUpperConfidenceBound(
                    self.gp,
                    self.sqrtbeta,
                    posterior_transform=posterior_transform,
                )
        # if different acquisitions should be used
        else:
            acq = qUpperConfidenceBound(
                self.gp,
                beta=self.sqrtbeta,
                posterior_transform=posterior_transform,
            )
        candidate, tt = optimize_acqf(
            acq_function=acq,
            bounds=(
                self.bounds
                if task == 0
                else self.bounds
                + torch.tensor(
                    [[self.obj.max_disturbance], [-self.obj.max_disturbance]] # max_disturbance is zero for LbSync (only shifts)
                )
            ),
            q=self.num_acq_samps[task],
            num_restarts=init_cond.size(0) if task == 0 else 1,
            raw_samples=512 if task != 0 else None,
            nonlinear_inequality_constraints=(
                [self.inequality_consts] if task == 0 else None
            ),
            batch_initial_conditions=init_cond if task == 0 else None,
            options={"maxiter": 25},
        )
        return candidate
    
    """
        Selects the next point using a SafeOpt-like strategy for 1D problems.
        Partitions the discretized input space into safe, maximizer, and expander
        sets, then selects a point to either maximize or expand the safe set.
        Args:
            posterior_transform (ScalarizedPosteriorTransform, optional): A posterior
                transform. Defaults to None.
        Returns:
            Tensor: The next point to evaluate.
        """
    def safeopt(self, posterior_transform=None):
        with torch.no_grad():
            grid = torch.linspace(self.bounds[0,0], self.bounds[1,0], 1000).unsqueeze(-1)
            self.gp.eval()
            preds = self.gp(torch.hstack((grid, torch.zeros(grid.size(0), 1))))
            mask = (preds.mean - self.sqrtbeta * preds.covariance_matrix.diag().sqrt() - self.norm_threshold) >= 0
            best_obs = self.train_targets[self.train_tasks == 0].argmax()
            maxi_mask = (preds.mean + self.sqrtbeta * preds.covariance_matrix.diag().sqrt() > best_obs) & mask
            expander_mask = torch.hstack([(mask[:-1] & ~mask[1:]) | (mask[1:] & ~mask[:-1]), torch.tensor(False)])
            expander_set = grid[expander_mask]
            print(f"expander:{expander_set.numel()}")
            maximizers = grid[maxi_mask]
            if torch.any(expander_mask):
                expmax, eidx = torch.max(preds.covariance_matrix.diag()[expander_mask], dim=0)
            else:
                expmax = -1
            if torch.any(maxi_mask):
                maxmax, maxmax_idx = torch.max(preds.covariance_matrix.diag()[maxi_mask], dim=0)
            else:
                maxmax = -1
        if expmax == -1 and maxmax == -1:
            safe_max = preds.covariance_matrix.diag()[mask].argmax()
            return grid[mask][safe_max]
        if expmax >= maxmax:
            return expander_set[eidx]
        else:
            return maximizers[maxmax_idx]
