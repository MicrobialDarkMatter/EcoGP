import torch
import pyro
import pyro.distributions as dist
import gpytorch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import tqdm

import wandb
import sys
import os
import warnings

from EcoGP.MultitaskVariationalStrategy import MultitaskVariationalStrategy
from EcoGP.likelihoods import DirichletMultinomialLikelihood, BernoulliLikelihood


class EcoGP(pyro.nn.PyroModule):
    """
    EcoGP model combining environmental and spatial Gaussian Processes.
    """
    def __init__(self,
                 n_latents_env=None,
                 n_variables=None,
                 n_inducing_points_env=None,
                 n_latents_spatial=None,
                 n_inducing_points_spatial=None,
                 unique_coordinates=None,
                 likelihood=None):
        super().__init__()

        self.n_latents_env = n_latents_env
        self.n_latents_spatial = n_latents_spatial

        assert self.n_latents_env is None or (isinstance(self.n_latents_env, int) and self.n_latents_env > 0), (
            warnings.warn("self.n_latents_env must be a positive integer or None", UserWarning))
        assert self.n_latents_spatial is None or (isinstance(self.n_latents_spatial, int) and self.n_latents_spatial > 0), (
            warnings.warn("self.n_latents_spatial must be a positive integer or None", UserWarning))

        if likelihood == "Bernoulli":
            self.likelihood = BernoulliLikelihood
        elif likelihood == "Dirichlet":
            self.likelihood = DirichletMultinomialLikelihood
        else:
            self.likelihood = likelihood

        if self.n_latents_env is not None:
            self.f = EnvironmentGP(n_latents=n_latents_env, n_variables=n_variables,
                                   n_inducing_points=n_inducing_points_env)

        if self.n_latents_spatial is not None:
            self.g = SpatialGP(n_latents=n_latents_spatial, unique_coordinates=unique_coordinates,
                               n_inducing_points=n_inducing_points_spatial)

    def model(self, X=None, Y=None, coords=None, traits=None, training=True):
        """
        Model specification for EcoGP.

        :param X: Environmental features as a tensor of shape [n_samples, n_variables].
        :param Y: Species occurrence/abundance data as a tensor of shape [n_samples, n_species].
        :param coords: Spatial coordinates as a tensor of shape [n_samples, 2] (latitude, longitude).
        :param traits: Species traits as a tensor of shape [n_species, n_traits].
        :param training: Boolean indicating if the model is in training mode.
        :return: None
        """
        pyro.module("model", self)

        n_samples = next(input_data.size(0) for input_data in (X, Y, coords) if input_data is not None)
        n_species = Y.size(1) if Y is not None else None
        n_traits = traits.size(1) if traits is not None else None

        samples_plate = pyro.plate(name="samples_plate", size=n_samples, dim=-2)
        species_plate = pyro.plate(name="species_plate", size=n_species, dim=-1)

        z = 0

        if self.n_latents_env is not None:
            latent_env_plate = pyro.plate("env_latents_plate_w", self.n_latents_env, dim=-2)
            f_dist = self.f.pyro_model(X, name_prefix="f_GP")

            # f independent across L latents
            with pyro.plate("L_plate", dim=-1):
                # Sample from latent function distribution
                f_samples = pyro.sample(".f(x)", f_dist)

            # Correcting shape-mismatch, which may occur using particles
            f_samples = f_samples if f_samples.shape == torch.Size([n_samples, self.n_latents_env]) else f_samples.mean(
                dim=0).reshape(n_samples, self.n_latents_env)

            if traits is not None:
                # If traits, w_loc is a linear combination of traits and gamma ~ N(0,1)
                traits_plate = pyro.plate(name="traits_plate", size=n_traits, dim=-1)
                with traits_plate, latent_env_plate:
                    gamma = pyro.sample("gamma", dist.Normal(loc=torch.zeros(self.n_latents_env, n_traits),
                                                             scale=torch.ones(self.n_latents_env, n_traits)))
                w_loc = pyro.deterministic("w_loc", (traits @ gamma.T).T)

                # gamma = pyro.param("gamma", torch.randn(self.n_latents_env, n_traits))
                # w_loc = (traits @ gamma.T).T
            else:
                w_loc = torch.zeros(self.n_latents_env, n_species)

            w_scale = torch.ones(self.n_latents_env, n_species)
            # w independent across environmental latents and species
            with species_plate, latent_env_plate:
                w = pyro.sample("w", dist.Normal(loc=w_loc, scale=w_scale))

            z = z + f_samples @ w

        varians = 1

        if self.n_latents_spatial is not None:
            g_dist = self.g.pyro_model(coords, name_prefix="g_GP")

            # g independent across M latents
            with pyro.plate("M_plate", dim=-1):
                # Sample from latent function distribution
                g_samples = pyro.sample(".g(coords)", g_dist)

            # Correcting shape-mismatch, which may occur using particles
            g_samples = g_samples if g_samples.shape == torch.Size(
                [n_samples, self.n_latents_spatial]) else g_samples.mean(dim=0).reshape(
                n_samples, self.n_latents_spatial)

            # v = pyro.param("v", torch.randn(self.n_latents_spatial, n_species))
            v_loc = torch.zeros(self.n_latents_spatial, n_species)
            v_scale = torch.ones(self.n_latents_spatial, n_species) * varians
            # v independent across spatial latents and species
            with species_plate, pyro.plate("spatial_latents_plate_v", self.n_latents_spatial, dim=-2):
                v = pyro.sample("v", dist.Normal(loc=v_loc, scale=v_scale))

            z = z + g_samples @ v

        # bias independent across species
        with species_plate:
            bias = pyro.sample("b", dist.Normal(loc=torch.zeros(n_species), scale=torch.ones(n_species) * varians))

        z = z + bias

        self.likelihood(z, Y, training, samples_plate, species_plate)

    def guide(self, X=None, Y=None, coords=None, traits=None, training=True):
        """
        Variational guide for EcoGP.

        :param X: Environmental features as a tensor of shape [n_samples, n_variables].
        :param Y: Species occurrence/abundance data as a tensor of shape [n_samples, n_species].
        :param coords: Spatial coordinates as a tensor of shape [n_samples, 2] (latitude, longitude).
        :param traits: Species traits as a tensor of shape [n_species, n_traits].
        :param training: Boolean indicating if the model is in training mode.
        :return: None
        """
        n_species = Y.size(1) if Y is not None else None
        n_traits = traits.size(1) if traits is not None else None

        species_plate = pyro.plate(name="species_plate", size=n_species, dim=-1)

        if self.n_latents_env is not None:
            latent_env_plate = pyro.plate("env_latents_plate_w", self.n_latents_env, dim=-2)
            # w_loc = pyro.param(
            #     "w_loc",
            #     torch.zeros(n_species, self.n_latents_env)
            # )
            #
            # # Shape: [n_species, n_latents_env, n_latents_env]
            # w_scale_tril = pyro.param(
            #     "w_scale_tril",
            #     0.1 * torch.eye(self.n_latents_env)
            #     .expand(n_species, self.n_latents_env, self.n_latents_env)
            #     .clone(),
            #     constraint=dist.constraints.lower_cholesky
            # )
            #
            # # -- CRITICAL PART: set dim=-1 so that species is the RIGHTMOST dimension.
            # with species_plate:
            #     # By default, MultivariateNormal(...):
            #     #   - batch shape = [n_species]
            #     #   - event shape = [n_latents_env]
            #     #
            #     # Placing the plate at dim=-1 forces the "event dimension" to be -2,
            #     # so physically the sample comes out [n_latents_env, n_species].
            #     w = pyro.sample(
            #         "w",
            #         dist.MultivariateNormal(w_loc, scale_tril=w_scale_tril)
            #     )

            if traits is not None:
                traits_plate = pyro.plate(name="traits_plate", size=n_traits, dim=-1)

                gamma_loc = pyro.param("gamma_loc", torch.zeros(self.n_latents_env, n_traits))
                gamma_scale = pyro.param("gamma_scale", 1.1 * torch.ones(self.n_latents_env, n_traits),
                                         constraint=dist.constraints.positive)
                with traits_plate, latent_env_plate:
                    # gamma = pyro.sample("gamma", dist.Normal(loc=torch.zeros(self.n_latents_env, n_traits), scale=torch.ones(self.n_latents_env, n_traits)))
                    gamma = pyro.sample("gamma", dist.Normal(loc=gamma_loc, scale=gamma_scale))
                # w_loc = pyro.deterministic("w_loc", (traits @ gamma.T).T)
                w_loc = (traits @ gamma.T).T

                # gamma = pyro.param("gamma", torch.randn(self.n_latents_env, n_traits))
                # w_loc = (traits @ gamma.T).T

            else:
                w_loc = pyro.param("w_loc", torch.zeros(self.n_latents_env, n_species))

            w_scale = pyro.param("w_scale", 1.1 * torch.ones(self.n_latents_env, n_species),
                                 constraint=dist.constraints.positive)

            with species_plate, latent_env_plate:
                w = pyro.sample("w", dist.Normal(loc=w_loc, scale=w_scale))

            # pyro.module(self.name_prefixes[i], self.gp_models[i])
            f_dist = self.f.pyro_guide(X, name_prefix="f_GP")
            # Use a plate here to mark conditional independencies
            with pyro.plate("L_plate", dim=-1):
                # Sample from latent function distribution
                f_samples = pyro.sample(".f(x)", f_dist)

        if self.n_latents_spatial is not None:
            g_dist = self.g.pyro_guide(coords, name_prefix="g_GP")  # TODO: BREAKER
            # Use a plate here to mark conditional independencies
            with pyro.plate("M_plate", dim=-1):
                # Sample from latent function distribution
                g_samples = pyro.sample(".g(coords)", g_dist)

            v_loc = pyro.param("v_loc", torch.zeros(self.n_latents_spatial, n_species))
            v_scale = pyro.param(
                "v_scale",
                1.1 * torch.ones(self.n_latents_spatial, n_species),
                constraint=dist.constraints.positive
            )

            with species_plate, pyro.plate("spatial_latents_plate_v", self.n_latents_spatial, dim=-2):
                v = pyro.sample("v", dist.Normal(loc=v_loc, scale=v_scale))

        # if self.traits:
        #     bias_loc = pyro.param("bias_loc", torch.zeros(n_species))
        #     bias_scale = pyro.param("bias_scale", torch.ones(n_species), constraint=dist.constraints.positive)
        #
        #     with species_plate:
        #         bias = pyro.sample("b", dist.Normal(loc=bias_loc, scale=bias_scale))

        bias_loc = pyro.param("bias_loc", torch.zeros(n_species))
        bias_scale = pyro.param("bias_scale", torch.ones(n_species) * 1.1, constraint=dist.constraints.positive)

        with species_plate:
            bias = pyro.sample("b", dist.Normal(loc=bias_loc, scale=bias_scale))

    def forward(self, X=None, Y=None, coords=None, traits=None, training=True):
        """
        Forward pass for point prediction, otherwise call self.model().

        :param X: Environmental features as a tensor of shape [n_samples, n_variables].
        :param Y: Species occurrence/abundance data as a tensor of shape [n_samples, n_species].
        :param coords: Spatial coordinates as a tensor of shape [n_samples, 2] (latitude, longitude).
        :param traits: Species traits as a tensor of shape [n_species, n_traits].
        :param training: Boolean indicating if the model is in training mode.
        :return: Predicted species occurrence/abundance as a tensor of shape [n_samples, n_species].
        """
        # Point prediction
        z = 0

        if self.n_latents_env is not None:
            f_samples = self.f.pyro_guide(X, name_prefix="f_GP").mean

            if traits is not None:
                w = (traits @ pyro.param("gamma_loc").T).T
                # w = (traits @ pyro.param("gamma").T).T
            else:
                w = pyro.param("w_loc")

            z = z + f_samples @ w

        if self.n_latents_spatial is not None:
            g_samples = self.g.pyro_guide(coords, name_prefix="g_GP").mean
            v = pyro.param("v_loc")

            z = z + g_samples @ v

        bias = pyro.param("bias_loc")

        z = z + bias

        if isinstance(self.likelihood, type(BernoulliLikelihood)):
            return dist.Bernoulli(logits=z).mean

        if isinstance(self.likelihood, type(DirichletMultinomialLikelihood)):
            return dist.Dirichlet(concentration=z).mean


class EnvironmentGP(gpytorch.models.ApproximateGP):
    """
    Environmental Gaussian Process model for EcoGP.
    """
    def __init__(self, n_latents, n_variables, n_inducing_points):
        self.n_latents = n_latents
        # Let's use a different set of inducing points for each latent function
        inducing_points = torch.randn(n_latents, n_inducing_points, n_variables)

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([n_latents])
        )

        variational_strategy = MultitaskVariationalStrategy(  # CustomVariationalStrategy
            base_variational_strategy=gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch, so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([n_latents]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=gpytorch.priors.NormalPrior(5, 5),#gpytorch.priors.GammaPrior(rate=1, concentration=5),
                batch_shape=torch.Size([n_latents]),
                ard_num_dims=n_variables,
            ),
            outputscale_prior=gpytorch.priors.NormalPrior(1, 1),#gpytorch.priors.GammaPrior(rate=1, concentration=2),
            batch_shape=torch.Size([n_latents])
        )

        # self.covar_module.base_kernel.lengthscale = torch.rand(n_latents, 1, n_variables)
        # print(self.covar_module.base_kernel.lengthscale)
        # self.covar_module.outputscale = torch.rand(n_latents, 1, 1)

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class HaversineRBFKernel(gpytorch.kernels.Kernel):
    """A GPyTorch kernel that computes the Haversine distance and applies an RBF transformation."""

    has_lengthscale = True  # Allows GPyTorch to learn the lengthscale

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x1, x2, diag=False, **params):
        """Compute the kernel matrix using Haversine distance with RBF transformation."""
        if diag:
            return torch.ones(1, x1.shape[-2])
        # Convert degrees to radians
        RADIUS = 6373  # Approximate radius of Earth in km

        # Convert degrees to radians
        lon1, lat1, lon2, lat2 = map(torch.deg2rad, (x1[:, :, 0], x1[:, :, 1], x2[:, :, 0], x2[:, :, 1]))

        # Compute differences
        dlon = lon2.unsqueeze(1) - lon1.unsqueeze(2)
        dlat = lat2.unsqueeze(1) - lat1.unsqueeze(2)

        # Haversine formula
        a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1.unsqueeze(2)) * torch.cos(lat2.unsqueeze(1)) * torch.sin(
            dlon / 2) ** 2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

        haversine_dist = RADIUS * c

        # Apply the RBF kernel
        rbf_kernel = torch.exp(-0.5 * (haversine_dist / self.lengthscale) ** 2)

        return rbf_kernel


class SpatialGP(gpytorch.models.ApproximateGP):
    """
    Spatial Gaussian Process model for EcoGP.
    """
    def __init__(self, n_latents, unique_coordinates, n_inducing_points):
        self.n_latents = n_latents
        num_coords = unique_coordinates.size(0)

        inducing_points = unique_coordinates[
                          torch.stack([torch.randperm(num_coords)[:n_inducing_points] for _ in range(self.n_latents)]),
                          :]

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([n_latents])
        )

        variational_strategy = MultitaskVariationalStrategy(  # CustomVariationalStrategy
            base_variational_strategy=gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch, so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([n_latents]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=gpytorch.priors.NormalPrior(5, 5),#gpytorch.priors.GammaPrior(rate=1, concentration=1),
                batch_shape=torch.Size([n_latents]),
                ard_num_dims=unique_coordinates.size(-1)
            ),
            outputscale_prior=gpytorch.priors.NormalPrior(1, 1),#gpytorch.priors.GammaPrior(rate=1, concentration=1),
            batch_shape=torch.Size([n_latents])
        )
        # self.covar_module.base_kernel.lengthscale = torch.rand(n_latents, 1, 1) * 5
        # self.covar_module.base_kernel.lengthscale = torch.ones(n_latents, 1, 1, requires_grad=False) * 3
        # self.covar_module.outputscale = torch.rand(n_latents, 1, 1)

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


