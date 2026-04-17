import torch

from development.HMSC.spatial.lambda_factor_loadings import lambdas_spatial_model  # TODO: Fix import when moved
from development.HMSC.spatial.eta_spatial_model import etas_spatial_model  # TODO: Fix import when moved


def spatial_model(batch, lengthscale):
    n_latents = len(lengthscale)

    etas = etas_spatial_model(batch, lengthscale)

    lambdas = lambdas_spatial_model(batch, n_latents)
    # lambdas = torch.zeros(dataset.get_Y().shape[1], n_latents)

    # Note: Lambda as h,j and not j,h!!!
    eps = etas @ lambdas.T

    return eps

