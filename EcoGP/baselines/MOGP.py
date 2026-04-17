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

from EcoGP.MultitaskVariationalStrategy import MultitaskVariationalStrategy


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

        self.n_latents_env = n_latents_env
        self.f = EnvironmentGP(n_latents=n_latents_env, n_variables=n_variables,
                               n_inducing_points=n_inducing_points_env)

    def model(self, X, Y, training):
        pyro.module("model", self)

        n_samples = next(input_data.size(0) for input_data in (X, Y) if input_data is not None)
        n_species = Y.size(1) if Y is not None else None

        samples_plate = pyro.plate(name="samples_plate", size=n_samples, dim=-2)
        species_plate = pyro.plate(name="species_plate", size=n_species, dim=-1)

        f_dist = self.f.pyro_model(X, name_prefix="f_GP")

        # Use a plate here to mark conditional independencies
        with pyro.plate(".data_plate", dim=-1):
            # Sample from latent function distribution
            f_samples = pyro.sample(".f(x)", f_dist)

        f_samples = f_samples if f_samples.shape == torch.Size([n_samples, self.n_latents_env]) else f_samples.mean(
            dim=0).reshape(n_samples, self.n_latents_env)

        with pyro.plate("species_plate-a", size=n_species, dim=-1):
            w = pyro.sample("w", dist.Normal(loc=torch.zeros(n_species, self.n_latents_env),
                                             scale=torch.ones(n_species, self.n_latents_env)).to_event(1))

        f_samples = f_samples if f_samples.shape == torch.Size([n_samples, self.n_latents_env]) else f_samples.mean(
            dim=0).reshape(n_samples, self.n_latents_env)

        z = f_samples @ w.T

        with species_plate:
            bias = pyro.sample("b", dist.Normal(loc=torch.zeros(n_species), scale=torch.ones(n_species)))

        z = z + bias

        with samples_plate, species_plate:
            pyro.sample("y", dist.Bernoulli(logits=z), obs=Y if training else None)

    def guide(self, X, Y, training):
        n_species = Y.size(1) if Y is not None else None

        species_plate = pyro.plate(name="species_plate", size=n_species, dim=-1)

        w_loc = pyro.param("w_loc", torch.zeros(n_species, self.n_latents_env))
        w_scale = pyro.param("w_scale", torch.ones(n_species, self.n_latents_env), constraint=dist.constraints.positive)

        with pyro.plate("species_plate-a", size=n_species, dim=-1):
            w = pyro.sample("w",
                            dist.Normal(loc=w_loc,
                                        scale=w_scale).to_event(1))

        # pyro.module(self.name_prefixes[i], self.gp_models[i])
        f_dist = self.f.pyro_guide(X, name_prefix="f_GP")
        # Use a plate here to mark conditional independencies
        with pyro.plate(".data_plate", dim=-1):
            # Sample from latent function distribution
            f_samples = pyro.sample(".f(x)", f_dist)


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
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([n_latents]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=gpytorch.priors.NormalPrior(loc=5, scale=100),
                batch_shape=torch.Size([n_latents]),
                #ard_num_dims=n_variables,
            ),
            outputscale_prior=gpytorch.priors.GammaPrior(rate=1, concentration=25),
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


if __name__ == "__main__":
    # LOAD DATA
    from torch.utils.data import DataLoader, random_split
    from EcoGP.DataLoad import DataLoad
    from EcoGP.DataSampler import DataSampler

    from EcoGP.misc.calculate_metrics_fast import calculate_metrics

    # from configs.config_clean_unique import config  # Import the config module
    # from configs.config_butterfly import config  # Import the config module
    # from configs.config_clean_unique import config
    from configs.config_toy import config

    # ARGUMENTS
    # environment = config["additive"]["environment"]
    # spatial = config["additive"]["spatial"]
    # traits = config["additive"]["traits"]

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
    split_pct = config["general"]["split_pct"]
    n_inducing_points_env = config["environmental"]["n_inducing_points"]
    n_inducing_points_spatial = config["spatial"]["n_inducing_points"]

    verbose = config["general"]["verbose"]
    presence_absence = config["data"]["presence_absence"]
    normalize_X = config["data"]["normalize_X"]
    likelihood = config["general"]["likelihood"]
    seed = config["general"]["seed"]

    # prevalence_threshold = config["data"]["prevalence_threshold"]
    # STOP ARGUMENTS

    res = {
        "ROC AUC": [],
        "PR AUC": [],
        "NLL": [],
        "MAE": [],

    }

    for seed in range(0, 5):
        pyro.clear_param_store()

        data = DataLoad(
            Y_path=y_path,
            X_path=x_path,
            coords_path=coords_path,
            traits_path=traits_path,
            device=device,
            normalize_X=normalize_X,
            total_counts_path="",
            presence_absence_Y=presence_absence,
            verbose=verbose
        )

        dataset = DataSampler(data)

        if coords_path:
            train_indices, validation_indices, test_indices = random_split(torch.arange(dataset.unique_coords.shape[0]),
                                                                           split_pct,
                                                                           generator=torch.Generator().manual_seed(
                                                                               seed))

            # Getting the spatial locations split into separate sets
            train_indices = dataset.coords_inverse_indicies[
                torch.isin(dataset.coords_inverse_indicies, torch.tensor(train_indices.indices))]
            validation_indices = dataset.coords_inverse_indicies[
                torch.isin(dataset.coords_inverse_indicies, torch.tensor(validation_indices.indices))]
            test_indices = dataset.coords_inverse_indicies[
                torch.isin(dataset.coords_inverse_indicies, torch.tensor(test_indices.indices))]

            train_dataset = torch.utils.data.Subset(dataset, train_indices)
            validation_dataset = torch.utils.data.Subset(dataset, validation_indices)
            test_dataset = torch.utils.data.Subset(dataset, test_indices)
        else:
            train_dataset, validation_dataset, test_dataset = random_split(dataset, split_pct,
                                                                           generator=torch.Generator().manual_seed(
                                                                               seed))

        # Make sure at least 1 species obserservations are present all splits
        # Can't make predictions for a species not present in training
        keep_y = (dataset.Y[train_dataset.indices].sum(dim=0) >= split_pct[0] * 10) & (
                dataset.Y[validation_dataset.indices].sum(dim=0) >= split_pct[1] * 10) & (
                         dataset.Y[test_dataset.indices].sum(dim=0) >= split_pct[2] * 10)
        dataset.Y = dataset.Y[:, keep_y]
        dataset.taxon_names = dataset.taxon_names[keep_y]
        dataset.n_species = dataset.Y.shape[1]
        if traits_path:
            dataset.traits = dataset.traits[keep_y, :]
        if verbose:
            print(f"Keeping {keep_y.sum().item()} taxons after at least one observation in each split")

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        n_tasks = dataset.n_species
        n_variables = dataset.n_env
        # n_traits = dataset.n_traits
        unique_coordinates = dataset.unique_coords if coords_path else None

        model = MicroGP(
            n_latents_env,
            n_variables,
            n_inducing_points_env,
            n_latents_spatial,
            n_inducing_points_spatial,
            unique_coordinates,
        ).to(device)

        optimizer = pyro.optim.Adam({"lr": lr})
        # elbo = pyro.infer.Trace_ELBO(num_particles=n_particles, vectorize_particles=True, retain_graph=True)

        from models.BetaTraceELBO import BetaTraceELBO

        elbo = BetaTraceELBO(beta=.5, num_particles=n_particles, vectorize_particles=True, retain_graph=True)

        svi = pyro.infer.SVI(model.model, model.guide, optimizer, elbo)

        model.train()
        training = True
        iterator = tqdm.tqdm(range(n_iter))
        for i in iterator:
            loss = 0
            for idx in train_dataloader:
                X, Y, _, _ = train_dataset.dataset.get_batch_data(idx)
                loss += svi.step(X, Y, training) / Y.nelement()

            iterator.set_postfix(loss=loss)

        test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

        y_prob_list = []
        y_test_list = []
        for idx in test_dataloader:
            X, Y, _, _ = test_dataset.dataset.get_batch_data(idx)
            training = False

            predictive = pyro.infer.Predictive(model.model, guide=model.guide, num_samples=200)
            y_prob = predictive(X, Y, training)["y"].mean(dim=0)
            y_prob_list.append(y_prob)

            y_test_list.append(Y)

        y_prob = torch.concat(y_prob_list)
        test_Y = torch.concat(y_test_list)
        del y_prob_list, y_test_list

        metrics = calculate_metrics(test_Y, y_prob)
        print(metrics)

        res["ROC AUC"].append(metrics["AUC"])
        res["NLL"].append(metrics["NLL"])
        res["MAE"].append(metrics["MAE"])
        res["PR AUC"].append(metrics["PR_AUC"])

    for key, value in res.items():
        print(key, torch.tensor(value).mean(), torch.tensor(value).std())

    print("Done")
