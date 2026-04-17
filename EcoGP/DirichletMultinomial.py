# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import numbers

import torch

from pyro.distributions import constraints
from pyro.distributions.torch import Dirichlet, Multinomial
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions.util import broadcast_shape


def _log_beta_1(alpha, value, is_sparse):
    if is_sparse:
        mask = value != 0
        value, alpha, mask = torch.broadcast_tensors(value, alpha, mask)
        result = torch.zeros_like(value, dtype=alpha.dtype)  # TODO: Included dtype
        value = value[mask]
        alpha = alpha[mask]
        result[mask] = (
            torch.lgamma(1 + value) + torch.lgamma(alpha) - torch.lgamma(value + alpha)
        )
        return result
    else:
        return (
            torch.lgamma(1 + value) + torch.lgamma(alpha) - torch.lgamma(value + alpha)
        )

class DirichletMultinomial(TorchDistribution):
    r"""
    Compound distribution comprising of a dirichlet-multinomial pair. The probability of
    classes (``probs`` for the :class:`~pyro.distributions.Multinomial` distribution)
    is unknown and randomly drawn from a :class:`~pyro.distributions.Dirichlet`
    distribution prior to a certain number of Categorical trials given by
    ``total_count``.

    :param float or torch.Tensor concentration: concentration parameter (alpha) for the
        Dirichlet distribution.
    :param int or torch.Tensor total_count: number of Categorical trials.
    :param bool is_sparse: Whether to assume value is mostly zero when computing
        :meth:`log_prob`, which can speed up computation when data is sparse.
    """

    arg_constraints = {
        "concentration": constraints.independent(constraints.positive, 1),
        "total_count": constraints.nonnegative_integer,
    }
    support = Multinomial.support

    def __init__(
        self, concentration, total_count=1, is_sparse=False, validate_args=None
    ):
        batch_shape = concentration.shape[:-1]
        event_shape = concentration.shape[-1:]
        if isinstance(total_count, numbers.Number):
            total_count = concentration.new_tensor(total_count)
        else:
            batch_shape = broadcast_shape(batch_shape, total_count.shape)
            concentration = concentration.expand(batch_shape + (-1,))
            total_count = total_count.expand(batch_shape)
        self._dirichlet = Dirichlet(concentration)
        self.total_count = total_count
        self.is_sparse = is_sparse
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @property
    def concentration(self):
        return self._dirichlet.concentration

    @staticmethod
    def infer_shapes(concentration, total_count=()):
        batch_shape = broadcast_shape(concentration[:-1], total_count)
        event_shape = concentration[-1:]
        return batch_shape, event_shape

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(DirichletMultinomial, _instance)
        batch_shape = torch.Size(batch_shape)
        new._dirichlet = self._dirichlet.expand(batch_shape)
        new.total_count = self.total_count.expand(batch_shape)
        new.is_sparse = self.is_sparse
        super(DirichletMultinomial, new).__init__(
            new._dirichlet.batch_shape, new._dirichlet.event_shape, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=()):
        probs = self._dirichlet.sample(sample_shape)
        total_count = int(self.total_count.max())
        if not self.total_count.min() == total_count:
            raise NotImplementedError(
                "Inhomogeneous total count not supported by `sample`."
            )
        return Multinomial(total_count, probs).sample()

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        alpha = self.concentration
        return _log_beta_1(alpha.sum(-1), value.sum(-1), self.is_sparse) - _log_beta_1(
            alpha, value, self.is_sparse
        ).sum(-1)

    @property
    def mean(self):
        return self._dirichlet.mean * self.total_count.unsqueeze(-1)

    @property
    def variance(self):
        n = self.total_count.unsqueeze(-1)
        alpha = self.concentration
        alpha_sum = self.concentration.sum(-1, keepdim=True)
        alpha_ratio = alpha / alpha_sum
        return n * alpha_ratio * (1 - alpha_ratio) * (n + alpha_sum) / (1 + alpha_sum)
