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

from models.MultitaskVariationalStrategy import MultitaskVariationalStrategy


class MicroGP(pyro.nn.PyroModule):
    def __init__(self,
                 n_latents_env=None,
                 n_variables=None,
                 n_inducing_points_env=None,
                 n_latents_spatial=None,
                 n_inducing_points_spatial=None,
                 unique_coordinates=None,
                 environment=True,
                 spatial=True,
                 traits=True):
        super().__init__()

        self.environment = environment
        self.spatial = spatial
        self.traits = traits

        assert self.environment + self.spatial + self.traits, f"Model cannot run without any components! {self.environment=}, {self.spatial =}, {self.traits=}"
        print(f"Running with components: {self.environment=}, {self.spatial=}, {self.traits=}")

        if self.environment:
            self.n_latents_env = n_latents_env
            self.f = EnvironmentGP(n_latents=n_latents_env, n_variables=n_variables,
                                   n_inducing_points=n_inducing_points_env)

        if self.spatial:
            self.n_latents_spatial = n_latents_spatial
            self.eta = SpatialGP(n_latents=n_latents_spatial, unique_coordinates=unique_coordinates,
                                 n_inducing_points=n_inducing_points_spatial)

    def model(self, batch):
        pyro.module("model", self)

        n_samples = batch.get("n_samples_batch")
        n_species = batch.get("n_species")
        n_traits = batch.get("n_traits")

        samples_plate = pyro.plate(name="samples_plate", size=n_samples, dim=-2)
        species_plate = pyro.plate(name="species_plate", size=n_species, dim=-1)

        z = 0

        if self.environment:
            latents_plate = pyro.plate(name="latents_plate", size=self.n_latents_env, dim=-2)

            f_dist = self.f.pyro_model(batch.get("X"), name_prefix="f_GP")

            # Use a plate here to mark conditional independencies
            with pyro.plate(".data_plate", dim=-1):
                # Sample from latent function distribution
                f_samples = pyro.sample(".f(x)", f_dist)

            f_samples = f_samples if f_samples.shape == torch.Size([n_samples, self.n_latents_env]) else f_samples.mean(
                dim=0).reshape(n_samples, self.n_latents_env)

            if self.traits:
                gamma = pyro.param("gamma", torch.zeros(self.n_latents_env, n_traits))
                #                w_loc = gamma @ batch.get("traits").T
                w_loc = batch.get("traits") @ gamma.T
            else:
                w_loc = torch.zeros(n_species, self.n_latents_env)

            # Insert a dimension of size 1 for the 'samples' axis => [1, n_species, n_latents_env]
            # w_loc = w_loc.unsqueeze(0)

            # scale is [n_species, n_latents_env], so also unsqueeze(0) => [1, n_species, n_latents_env]
            # scale = torch.ones(n_species, n_latents_env).unsqueeze(0)

            #            with pyro.plate(name="samples_plate-a", size=n_samples):
            #          with pyro.plate(name="species_plate-a", size=n_species):# species_plate, latents_plate:

            with pyro.plate("species_plate-a", size=n_species, dim=-1):
                w = pyro.sample("w", dist.Normal(loc=w_loc, scale=torch.ones_like(w_loc)).to_event(1))

            f_samples = f_samples if f_samples.shape == torch.Size([n_samples, self.n_latents_env]) else f_samples.mean(
                dim=0).reshape(n_samples, self.n_latents_env)

            z = z + f_samples @ w.squeeze().T

        if self.spatial:
            eta_dist = self.eta.pyro_model(batch.get("coords"), name_prefix="eta_GP")

            with pyro.plate(".eta_data_plate", dim=-1):
                # Sample from latent function distribution
                eta_samples = pyro.sample(".eta(coords)", eta_dist)

            eta_samples = eta_samples if eta_samples.shape == torch.Size(
                [batch["n_locs_batch"], self.n_latents_spatial]) else eta_samples.mean(dim=0).reshape(
                batch["n_locs_batch"], self.n_latents_spatial)
            eta_samples = eta_samples[batch["batch_inverse"]]

            v = pyro.param("v", torch.randn(self.n_latents_spatial, n_species))

            z = z + eta_samples @ v

        with species_plate:
            bias = pyro.sample("b", dist.Normal(loc=torch.zeros(n_species), scale=torch.ones(n_species)))

        z = z + bias

        pyro.deterministic("z", z)

        with samples_plate, species_plate:
            pyro.sample("y", dist.Bernoulli(logits=z), obs=batch.get("Y") if batch.get("training", True) else None)

    def guide(self, batch):
        n_species = batch.get("n_species")
        species_plate = pyro.plate(name="species_plate", size=n_species, dim=-1)

        if self.environment:
            latents_plate = pyro.plate(name="latents_plate", size=self.n_latents_env, dim=-2)

            # if self.traits:
            #     gamma = pyro.param("gamma", torch.zeros(self.n_latents_env, n_traits))
            #     w_loc = gamma @ batch.get("traits").T
            # else:

            w_loc = pyro.param(
                "w_loc",
                torch.zeros(n_species, self.n_latents_env)
            )

            # Shape: [n_species, n_latents_env, n_latents_env]
            w_scale_tril = pyro.param(
                "w_scale_tril",
                0.1 * torch.eye(self.n_latents_env)
                .expand(n_species, self.n_latents_env, self.n_latents_env)
                .clone(),
                constraint=dist.constraints.lower_cholesky
            )

            # -- CRITICAL PART: set dim=-1 so that species is the RIGHTMOST dimension.
            with pyro.plate("species_plate-a", n_species, dim=-1):
                # By default, MultivariateNormal(...):
                #   - batch shape = [n_species]
                #   - event shape = [n_latents_env]
                #
                # Placing the plate at dim=-1 forces the "event dimension" to be -2,
                # so physically the sample comes out [n_latents_env, n_species].
                w = pyro.sample(
                    "w",
                    dist.MultivariateNormal(w_loc, scale_tril=w_scale_tril)
                )

            # w_loc = pyro.param("w_loc", torch.zeros(n_species, self.n_latents_env))
            # w_scale = pyro.param("w_scale", torch.ones(n_species, self.n_latents_env), constraint=dist.constraints.positive)

            # with pyro.plate("species_plate-a", size=n_species, dim=-1):
            #     w = pyro.sample("w",
            #                     dist.Normal(loc=w_loc,
            #                                 scale=w_scale).to_event(1))

            # pyro.module(self.name_prefixes[i], self.gp_models[i])
            f_dist = self.f.pyro_guide(batch.get("X"), name_prefix="f_GP")
            # Use a plate here to mark conditional independencies
            with pyro.plate(".data_plate", dim=-1):
                # Sample from latent function distribution
                f_samples = pyro.sample(".f(x)", f_dist)

        if self.spatial:
            eta_dist = self.eta.pyro_guide(batch.get("coords"), name_prefix="eta_GP")  # TODO: BREAKER
            # Use a plate here to mark conditional independencies
            with pyro.plate(".eta_data_plate", dim=-1):
                # Sample from latent function distribution
                eta_samples = pyro.sample(".eta(coords)", eta_dist)

        # if self.traits:
        #     bias_loc = pyro.param("bias_loc", torch.zeros(n_species))
        #     bias_scale = pyro.param("bias_scale", torch.ones(n_species), constraint=dist.constraints.positive)
        #
        #     with species_plate:
        #         bias = pyro.sample("b", dist.Normal(loc=bias_loc, scale=bias_scale))

        bias_loc = pyro.param("bias_loc", torch.zeros(n_species))
        bias_scale = pyro.param("bias_scale", torch.ones(n_species), constraint=dist.constraints.positive)

        with species_plate:
            bias = pyro.sample("b", dist.Normal(loc=bias_loc, scale=bias_scale))

    def forward(self, x):
        ...


class EnvironmentGP(gpytorch.models.ApproximateGP):
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
        self.mean_module = gpytorch.means.ConstantMean(prior=gpytorch.priors.NormalPrior(loc=-3, scale=1), batch_shape=torch.Size([n_latents]))
        # self.mean_module = gpytorch.means.LinearMean(input_size=n_variables, batch_shape=torch.Size([n_latents]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=gpytorch.priors.NormalPrior(loc=5, scale=10),
                batch_shape=torch.Size([n_latents]),
                ard_num_dims=n_variables,
            ),
            outputscale_prior=gpytorch.priors.GammaPrior(rate=1, concentration=5),
            batch_shape=torch.Size([n_latents])
        )

        # self.covar_module.base_kernel.lengthscale = torch.rand(n_latents, 1, n_variables)
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
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([n_latents]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(#HaversineRBFKernel(  # gpytorch.kernels.RBFKernel(#HaversineRBFKernel(  # CustomSpatialKernel(#
                lengthscale_prior=gpytorch.priors.NormalPrior(loc=5, scale=25),
                batch_shape=torch.Size([n_latents]),
            ),
            outputscale_prior=gpytorch.priors.GammaPrior(rate=1, concentration=1),
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


if __name__ == "__main__":
    # LOAD DATA
    from torch.utils.data import DataLoader, random_split
    from DataSampler import DataSampler

    from misc.calculate_metrics import calculate_metrics, precision_at_k

    from sklearn import metrics

    # Add the parent directory (or any other directory where the config module is located) to the Python path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../configs')))

    from configs.config import config  # Import the config module

    # ARGUMENTS
    environment = config["additive"]["environment"]
    spatial = config["additive"]["spatial"]
    traits = config["additive"]["traits"]

    x_path = config["data"]["X_path"]
    y_path = config["data"]["Y_path"]
    coords_path = config["data"]["coords_path"]
    traits_path = config["data"]["traits_path"]

    n_latents_env = config["environmental"]["n_latents"]
    n_latents_spatial = config["spatial"]["n_latents"]
    n_iter = config["general"]["n_iter"]
    n_particles = config["general"]["n_particles"]
    device = config["general"]["device"]
    lr = config["general"]["lr"]
    batch_size = config["general"]["batch_size"]
    train_pct = config["general"]["train_pct"]
    n_inducing_points_env = config["environmental"]["n_inducing_points"]
    n_inducing_points_spatial = config["spatial"]["n_inducing_points"]

    prevalence_threshold = config["data"]["prevalence_threshold"]

    save_model_path = config["general"]["save_model_path"]
    # STOP ARGUMENTS

    dataset = DataSampler(
        Y_path=y_path,
        X_path=x_path,
        coords_path=coords_path,
        traits_path=traits_path,
        device=device,
        normalize_X=True,
        prevalence_threshold=prevalence_threshold)

    if spatial:
        train_indices, test_indices = random_split(torch.arange(dataset.unique_coords.shape[0]),
                                                   [train_pct, 1 - train_pct],
                                                   generator=torch.Generator().manual_seed(42))

        # Getting the spatial locations split into separate sets
        train_indices = dataset.coords_inverse_indicies[
            torch.isin(dataset.coords_inverse_indicies, torch.tensor(train_indices.indices))]
        test_indices = dataset.coords_inverse_indicies[
            torch.isin(dataset.coords_inverse_indicies, torch.tensor(test_indices.indices))]

        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
    else:
        # dataloader = DataLoader(dataset=dataset, batch_size=_batch_size, shuffle=True)
        train_size = int(train_pct * len(dataset))
        test_size = len(dataset) - train_size

        train_dataset, test_dataset = random_split(dataset, [train_size, test_size],
                                                   generator=torch.Generator().manual_seed(42))

    # Make sure at least 10 species obserservations are present in each subset of the data
    keep_y = (dataset.Y[train_dataset.indices].sum(dim=0) >= 10) & (
                dataset.Y[test_dataset.indices].sum(dim=0) >= 10)
    dataset.Y = dataset.Y[:, keep_y]
    dataset.Y_cols_species = dataset.Y_cols_species[keep_y]
    dataset.n_species = dataset.Y.shape[1]
    if traits_path:
        dataset.traits = dataset.traits[keep_y, :]

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    n_tasks = dataset.n_species
    n_variables = dataset.n_env
    # n_traits = dataset.n_traits
    unique_coordinates = dataset.unique_coords[
        dataset.get_dist_idx_reverse(train_dataset.indices)[0]] if spatial else None

    model = MicroGP(
        n_latents_env,
        n_variables,
        n_inducing_points_env,
        n_latents_spatial,
        n_inducing_points_spatial,
        unique_coordinates,
        environment=environment,
        spatial=spatial,
        traits=traits
    ).to(device)

    optimizer = pyro.optim.Adam({"lr": lr})
    # elbo = pyro.infer.Trace_ELBO(num_particles=n_particles, vectorize_particles=True, retain_graph=True)

    from models.BetaTraceELBO import BetaTraceELBO

    elbo = BetaTraceELBO(beta=.5, num_particles=n_particles, vectorize_particles=True, retain_graph=True)

    svi = pyro.infer.SVI(model.model, model.guide, optimizer, elbo)

    model.train()

    iterator = tqdm.tqdm(range(n_iter))
    for i in iterator:
        loss = 0
        for idx in train_dataloader:
            batch = train_dataset.dataset.get_batch_data(idx)
            loss += svi.step(batch) / batch.get("Y").nelement()

        iterator.set_postfix(loss=loss)


    # Save model
    if save_model_path:
        torch.save(model, os.path.join(save_model_path, "model.pt"))
        pyro.get_param_store().save(os.path.join(save_model_path, "param_store.pt"))# f"../results/saved_models/param_store.pt"
        torch.save(dataset, os.path.join(save_model_path, "dataset.pt"))

    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    y_prob_list = []
    y_test_list = []
    for idx in test_dataloader:
        batch = test_dataset.dataset.get_batch_data(idx)
        batch["training"] = False
        batch["do_spatial"] = True

        predictive = pyro.infer.Predictive(model.model, guide=model.guide, num_samples=50)
        y_prob = predictive(batch)["y"].mean(dim=0)
        y_prob_list.append(y_prob)

        y_test_list.append(batch.get("Y"))

    y_prob = torch.concat(y_prob_list)
    test_Y = torch.concat(y_test_list)
    del y_prob_list, y_test_list

    test_Y_prev = test_Y.sum(dim=0) / test_Y.shape[0]

    metric_results = calculate_metrics(test_Y, y_prob)

    print(metric_results)
