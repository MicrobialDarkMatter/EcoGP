#!/usr/bin/env python3

from gpytorch.distributions import MultitaskMultivariateNormal
from gpytorch.module import Module
from gpytorch.variational._variational_strategy import _VariationalStrategy


class MultitaskVariationalStrategy(_VariationalStrategy):
    """
    The base variational strategy is assumed to operate on a batch of GPs. One of the batch
    dimensions corresponds to the multiple tasks.

    :param ~gpytorch.variational.VariationalStrategy base_variational_strategy: Base variational strategy
    :param int num_tasks: Number of tasks. Should correspond to the batch size of task_dim.
    :param int task_dim: (Default: -1) Which batch dimension is the task dimension
    """

    def __init__(self, base_variational_strategy, task_dim=-1):
        Module.__init__(self)
        self.base_variational_strategy = base_variational_strategy
        self.task_dim = task_dim

    @property
    def prior_distribution(self):
        return self.base_variational_strategy.prior_distribution

    @property
    def variational_distribution(self):
        return self.base_variational_strategy.variational_distribution

    @property
    def variational_params_initialized(self):
        return self.base_variational_strategy.variational_params_initialized

    def kl_divergence(self):
        return super().kl_divergence().sum(dim=-1)

    def __call__(self, x, task_indices=None, prior=False, **kwargs):
        function_dist = self.base_variational_strategy(x, prior=prior, **kwargs)

        function_dist = MultitaskMultivariateNormal.from_batch_mvn(function_dist, task_dim=self.task_dim)

        return function_dist
