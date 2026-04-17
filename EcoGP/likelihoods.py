import torch
import pyro
import pyro.distributions as dist

from EcoGP.DirichletMultinomial import DirichletMultinomial


def DirichletMultinomialLikelihood(z, Y, training, samples_plate, species_plate):
    """
    Dirichlet-Multinomial likelihood for modeling count data.

    :param z: Sum from LMC of environmental and spatial components as a tensor of shape [n_samples, n_species].
    :param Y: Species occurrence/abundance data as a tensor of shape [n_samples, n_species].
    :param training: Boolean to indicate training or prediction mode, thus "masking" the observations during prediction
    :param samples_plate: pyro.plate for samples
    :param species_plate: pyro.plate for species
    :return: None
    """
    z = torch.nn.functional.softplus(z) + 1e-6

    pyro.deterministic("z", z)

    if training:
        with samples_plate:
            # IMPORTANT: no species_plate here; species is the Dirichlet EVENT dim
            pyro.sample("y", DirichletMultinomial(concentration=z, total_count=Y.sum(dim=1), is_sparse=True),
                        obs=Y)
    else:
        # No plate for predictive !!!
        pyro.sample("y", dist.Dirichlet(concentration=z), obs=None)


def BernoulliLikelihood(z, Y, training, samples_plate, species_plate):
    """
    Bernoulli likelihood for modeling presence-absence data.

    :param z: Sum from LMC of environmental and spatial components as a tensor of shape [n_samples, n_species].
    :param Y: Species occurrence/abundance data as a tensor of shape [n_samples, n_species].
    :param training: Boolean to indicate training or prediction mode, thus "masking" the observations during prediction
    :param samples_plate: pyro.plate for samples
    :param species_plate: pyro.plate for species
    :return: None
    """
    pyro.deterministic("z", z)

    with samples_plate, species_plate:
        pyro.sample("y", dist.Bernoulli(logits=z),
                    obs=Y.bool().float() if training else None)
