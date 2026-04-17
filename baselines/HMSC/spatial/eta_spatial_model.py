import inspect

import torch
import warnings

import pyro
import pyro.distributions as dist
import torch.distributions.constraints as constraints
from development.HMSC.spatial.eta_covariance_matrix import \
    get_eta_covariance_matrix  # TODO: Fix import when moved


def etas_spatial_model(batch, lengthscale):
    """

    :param device:
    :param n_latents:
    :param eta_covariance:
    :param eta_inverse_indicies:
    :return:
    """
    device = batch.get("device")

    n_latents = len(lengthscale)
    inverse_indices = batch.get("batch_inverse")
    n_locs_batch = batch.get("n_locs_batch")

    distance_matrix = batch.get("dist").to(device)

    if n_locs_batch < 10:
        warnings.warn(f'Less than 10 unique sites in {inspect.currentframe().f_code.co_name}')

    eta_lengthscale = pyro.param("eta_lengthscale", lengthscale.to(device), constraint=constraints.positive)

    eta_covariance = get_eta_covariance_matrix(distance_matrix, eta_lengthscale)

    with pyro.plate("etas_plate", n_latents, dim=-1):
        etas = pyro.sample(f"etas", dist.MultivariateNormal(loc=torch.zeros(n_latents, n_locs_batch, device=device),
                                                            covariance_matrix=eta_covariance).to_event(0))

    etas = etas.T[inverse_indices]

    return etas

