import warnings

import pyro
import pyro.distributions as dist
import torch
import torch.distributions.constraints as constraints
from torch.utils.data import DataLoader, random_split
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.optim import Adam
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import KFold
import tqdm

import gpytorch

from EcoGP.MultitaskVariationalStrategy import MultitaskVariationalStrategy

from EcoGP.DataLoad import DataLoad
from EcoGP.DataSampler import DataSampler
from spatial.eta_covariance_matrix import get_eta_covariance_matrix


class HaversineRBFKernel(gpytorch.kernels.Kernel):
    """A GPyTorch kernel that computes the Haversine distance and applies an RBF transformation."""

    #is_stationary = True
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
        # lon1, lat1, lon2, lat2 = map(torch.deg2rad, (x1[:, :, 0], x1[:, :, 1], x2[:, :, 0], x2[:, :, 1]))

        lon1 = torch.deg2rad(x1[:, :, 0])
        lat1 = torch.deg2rad(x1[:, :, 1])
        lon2 = torch.deg2rad(x2[:, :, 0])
        lat2 = torch.deg2rad(x2[:, :, 1])
        # lon1 = lon2
        # lat1 = lat2


        # Compute differences
        dlon = lon2 - lon1.unsqueeze(-1)  # Shape: (N, M, K)
        dlat = lat2 - lat1.unsqueeze(-1)  # Shape: (N, M, K)

        # Haversine formula
        a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1.unsqueeze(-1)) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

        haversine_dist = RADIUS * c

        # Apply the RBF kernel
        rbf_kernel = torch.exp(-0.5 * (haversine_dist / self.lengthscale) ** 2)

        return rbf_kernel



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
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([n_latents]))
        self.covar_module = gpytorch.kernels.RBFKernel(
            lengthscale_prior=gpytorch.priors.GammaPrior(rate=1, concentration=5),
            batch_shape=torch.Size([n_latents]),
            ard_num_dims=n_variables,
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
                self, inducing_points, variational_distribution, learn_inducing_locations=False
            ),
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch, so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([n_latents]))
        self.covar_module = gpytorch.kernels.RBFKernel(
            # HaversineRBFKernel(  # gpytorch.kernels.RBFKernel(#HaversineRBFKernel(  # CustomSpatialKernel(#
            lengthscale_prior=gpytorch.priors.GammaPrior(rate=1, concentration=5),
            batch_shape=torch.Size([n_latents]),
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


class HMSC_GP(pyro.nn.PyroModule):
    def __init__(self, n_latents, unique_coordinates, n_inducing_points):
        super().__init__()

        self.n_spatial = n_latents
        self.eta = SpatialGP(n_latents=n_latents, unique_coordinates=unique_coordinates, n_inducing_points=n_inducing_points)

    def model(self, X=None, Y=None, coords=None, traits=None, training=True, device=torch.device("cpu"), likelihood=None):
        pyro.module("model", self)

        n_species = Y.size(1) if Y is not None else None
        n_traits = traits.size(1) if traits is not None else None
        n_env = X.size(1) if X is not None else None

        env_plate = pyro.plate(name="env_plate", size=n_env, dim=-2)
        trait_plate = pyro.plate(name="trait_plate", size=n_traits, dim=-2)

        if traits is not None:
            with trait_plate:
                gamma = pyro.sample("gamma", dist.Normal(loc=torch.zeros(n_traits, n_env, device=device),
                                                         scale=torch.ones(n_traits, n_env, device=device)))

            beta_loc = (traits @ gamma).T
        else:
            beta_loc = torch.zeros(n_env, n_species, device=device)

        with env_plate:
            # beta = pyro.sample("beta", dist.MultivariateNormal(loc=torch.zeros(n_env, n_species, device=device),
            #                                                    covariance_matrix=torch.tile(V, (n_env, 1, 1))))
            beta = pyro.sample("beta", dist.Normal(loc=beta_loc,
                                                   scale=torch.ones(n_env, n_species, device=device)))

        # Define the latent means (L)
        L = X @ beta

        if coords is not None:
            eps = self.hmsc_spatial_model(coords, n_species, device)
            L += eps.squeeze()

        # Changes model likelihood for final step
        self.model_likelihood(L, X, Y, coords, device, training, likelihood)


    def hmsc_spatial_model(self, coords, n_species, device):
        lengthscale = torch.ones(self.n_spatial, device=device)  # TODO: Add such that it can be modified!

        n_latents = self.n_spatial

        # ### ETAS ### #
        eta_dist = self.eta.pyro_model(coords, name_prefix="eta_GP")

        with pyro.plate("M_plate", dim=-1):
            # Sample from latent function distribution
            etas = pyro.sample(".eta(coords)", eta_dist)

        # ### LAMBDAS ### #
        latent_plate = pyro.plate("latent_plate", n_latents, dim=-1)
        species_plate = pyro.plate("_species_plate", n_species, dim=-2)

        # TODO: Possibly hyperparameters and not learnable
        v = torch.tensor(3.0,
                         device=device)  # pyro.param("v", torch.tensor(1.0), constraint=constraints.positive).to(device)
        a1 = torch.tensor(50.0,
                          device=device)  # pyro.param("a1", torch.tensor(1.0), constraint=constraints.positive).to(device)
        a2 = torch.tensor(50.0,
                          device=device)  # pyro.param("a2", torch.tensor(1.0), constraint=constraints.positive).to(device)

        # Tau
        delta1 = pyro.sample(f"delta_1", dist.Gamma(a1, 1.)).reshape(-1)
        if n_latents > 1:
            with pyro.plate("delta2_plate", n_latents - 1, dim=-1):
                delta2_up = pyro.sample(f"delta_2_up",
                                        dist.Gamma(concentration=torch.full((n_latents - 1,), a2, device=device),
                                                   rate=torch.full((n_latents - 1,), 1., device=device))).reshape(-1)

            tau = torch.cumprod(torch.cat((delta1, delta2_up), dim=0), dim=0)
        else:
            tau = delta1

        with species_plate, latent_plate:
            phi = pyro.sample("phi", dist.Gamma(v / 2, v / 2))

            lambdas = pyro.sample("lambdas",
                                  dist.Normal(torch.zeros(n_species, n_latents, device=device), phi ** -1 * tau ** -1))

        # Note: Lambda as h,j and not j,h!!!
        eps = etas @ lambdas.T

        return eps


    def guide(self, X=None, Y=None, coords=None, traits=None, training=True, device=torch.device("cpu"), likelihood=None):
        n_species = Y.size(1) if Y is not None else None
        n_traits = traits.size(1) if traits is not None else None
        n_env = X.size(1) if X is not None else None

        # species_plate = pyro.plate("species_plate", n_species, dim=-1)
        env_plate = pyro.plate("env_plate", n_env, dim=-2)
        trait_plate = pyro.plate(name="trait_plate", size=n_traits, dim=-2)

        # with species_plate:
        #     sigma = pyro.param("sigma", torch.ones(n_species, device=device), constraint=constraints.positive)

        if traits is not None:
            gamma_loc = pyro.param("gamma_loc", torch.zeros(n_traits, n_env, device=device))
            gamma_scale = pyro.param("gamma_scale", torch.ones(n_traits, n_env, device=device),
                                     constraint=constraints.positive)

            with trait_plate:
                gamma = pyro.sample("gamma", dist.Normal(loc=gamma_loc, scale=gamma_scale))

        beta_loc = pyro.param("beta_loc", torch.zeros(n_env, n_species, device=device))
        beta_scale = pyro.param("beta_scale", torch.ones(n_env, n_species, device=device), constraint=constraints.positive)

        with env_plate:
            beta = pyro.sample("beta", dist.Normal(loc=beta_loc, scale=beta_scale))

        if coords is not None:
            # eta
            n_latents = self.n_spatial

            eta_dist = self.eta.pyro_guide(coords, name_prefix="eta_GP")  # TODO: BREAKER
            # Use a plate here to mark conditional independencies
            with pyro.plate("M_plate", dim=-1):
                # Sample from latent function distribution
                etas = pyro.sample(".eta(coords)", eta_dist)

            # lambda
            latent_plate = pyro.plate("latent_plate", n_latents, dim=-1)
            _species_plate = pyro.plate("_species_plate", n_species, dim=-2)

            ## Tau
            delta1_loc = pyro.param("delta_1_loc", torch.tensor(0., device=device))
            delta1_scale = pyro.param("delta_1_scale", torch.ones(1, device=device), constraint=constraints.positive)
            delta1 = pyro.sample(f"delta_1", dist.TransformedDistribution(dist.Normal(delta1_loc, delta1_scale),
                                                                          dist.transforms.SoftplusTransform()))  # unsqueeze to move from scalar to vector

            if n_latents > 1:
                with pyro.plate("delta2_plate", n_latents - 1, dim=-1):
                    delta2_up_loc = pyro.param("delta_2_up_loc", torch.full((n_latents - 1,), 0., device=device))
                    delta2_up_scale = pyro.param("delta_2_up_scale", torch.full((n_latents - 1,), 1., device=device),
                                                 constraint=constraints.positive)
                    delta2_up = pyro.sample(f"delta_2_up",
                                            dist.TransformedDistribution(dist.Normal(delta2_up_loc, delta2_up_scale),
                                                                         dist.transforms.SoftplusTransform()))

                tau = torch.cumprod(torch.cat((delta1, delta2_up), dim=0), dim=0)
            else:
                tau = delta1

            with _species_plate, latent_plate:
                phi_loc = pyro.param("phi_loc", torch.ones(n_species, n_latents, device=device))
                phi_scale = pyro.param("phi_scale", torch.ones(n_species, n_latents, device=device),
                                       constraint=constraints.positive)
                phi = pyro.sample("phi", dist.TransformedDistribution(dist.Normal(phi_loc, phi_scale),
                                                                      dist.transforms.SoftplusTransform()))

                lambdas_loc = pyro.param("lambdas_loc", torch.zeros(n_species, n_latents, device=device))
                lambdas_scale = pyro.param("lambdas_scale", torch.ones(n_species, n_latents, device=device),
                                           constraint=constraints.positive)
                lambdas = pyro.sample("lambdas", dist.Normal(lambdas_loc, lambdas_scale))

        self.guide_likelihood(n_species, device, likelihood)


    def model_likelihood(self, L, X, Y, coords, device, training, likelihood):

        n_samples = next(input_data.size(0) for input_data in (X, Y, coords) if input_data is not None)
        n_species = Y.size(1) if Y is not None else None

        samples_plate = pyro.plate(name="samples_plate", size=n_samples, dim=-2)
        species_plate = pyro.plate(name="species_plate", size=n_species, dim=-1)

        match likelihood:
            case "normal":
                with species_plate:
                    sigma = pyro.param("sigma", torch.ones(n_species, device=device), constraint=constraints.positive)

                # Define the likelihood of the observed data
                with samples_plate, species_plate:
                    pyro.sample(f"y", dist.Normal(L, sigma), obs=Y if training else None)

            case "bernoulli":
                with samples_plate, species_plate:
                    pyro.sample(f"y", dist.Bernoulli(logits=L), obs=Y.bool().float() if training else None)

            case "dirichlet_multinomial":
                L = torch.nn.Softplus()(L)

                L += 1e-8  # Added due to numerical error when 0.0

                with samples_plate:
                    pyro.sample(f"y", dist.DirichletMultinomial(L, total_count=Y.sum(dim=1), is_sparse=True),
                                obs=Y if training else None)
            case _:
                warnings.warn("Likelihood not defined!")


    def guide_likelihood(self, n_species, device, likelihood):
        match likelihood:
            case "normal":
                species_plate = pyro.plate("species_plate", n_species, dim=-1)

                with species_plate:
                    sigma = pyro.param("sigma", torch.ones(n_species, device=device), constraint=constraints.positive)
            case "bernoulli":
                pass
            case "dirichlet_multinomial":
                pass
            case _:
                warnings.warn("Likelihood not defined!")


def train_svi(train_dataset, train_dataloader, epoch, model, guide, likelihood, optimizer, verbose):
    pyro.clear_param_store()

    # Set up the inference algorithm
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO(retain_graph=True))

    # Store the loss values
    loss_list = []

    training = True
    device = torch.device("cpu")

    # Perform inference
    for step in tqdm.tqdm(range(epoch)):
        # print(f"Starting step {step} of {epoch}")
        loss = 0
        for idx in train_dataloader:
            X, Y, coords, traits = train_dataset.dataset.get_batch_data(idx)
            loss += svi.step(X, Y, coords, traits, training, device, likelihood)
        loss_list.append(loss)

    # if verbose:
    #     print("Param store:")
    #     print(pyro.get_param_store().keys())

    # if verbose:
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(loss_list)
    #     plt.xlabel("Epoch")
    #     plt.ylabel("Loss")
    #     plt.title("Loss Curve")
    #     plt.show()


# def train_svi_cv(k_fold, train_dataset, batch_size, epoch, model, guide, likelihood, optimizer, verbose):
#     print("Depricated")
#     kf = KFold(n_splits=k_fold, shuffle=True)
#
#     # Loop through each fold
#     for fold, (train_idx, test_idx) in enumerate(kf.split(train_dataset)):
#         print(f"\n~~~~~ Fold {fold + 1} ~~~~~")
#
#         # Define the data loaders for the current fold
#         train_dataloader = DataLoader(
#             dataset=train_dataset,
#             batch_size=batch_size,
#             sampler=torch.utils.data.SubsetRandomSampler(train_idx),
#         )
#
#         train_svi(
#             train_dataset=train_dataset,
#             train_dataloader=train_dataloader,
#             epoch=epoch,
#             model=model,
#             guide=guide,
#             likelihood=likelihood,
#             optimizer=optimizer,
#             verbose=verbose,
#         )
#
#         # Evaluate
#         test_idx = train_dataset.indices
#         # validation_data = dataset.get_batch_data(validation_idx)
#         test_data = train_dataset.dataset.get_batch_data(test_idx)
#         test_data["training"] = False
#
#         predictive = Predictive(model, guide=guide, num_samples=50)
#
#         predict = predictive(test_data, likelihood)["y"].mean(dim=0)
#
#         auc_per_species = [
#             metrics.roc_auc_score(test_data.get("Y")[:, i].bool().int(), predict[:, i]) if not all(
#                 test_data.get("Y")[:, i] == 0) else float("nan") for i in
#             range(test_data.get("Y").shape[1])
#         ]
#
#         auc = torch.tensor(auc_per_species)
#         means_tensor = auc[~torch.isnan(auc)]
#
#         if True:  # Metrics
#             above_average = (means_tensor > 0.5).sum().item()
#             below_average = (means_tensor <= 0.5).sum().item()
#
#             pct_good = above_average / (above_average + below_average)
#             print(f"Species ROC above 50%: {pct_good * 100:.2f}%")
#
#             print(f"Species ROC above 50% * average: {pct_good * means_tensor.mean():.4f}")


def learn_model():
    # from configs.config_hmsc import config  # TODO: Set config

    # from configs.config_clean_unique import config  # TODO: Set config
    # from configs.config_butterfly import config  # Import the config module
    # from configs.config_central_park import config
    from configs.config_toy import config

    from EcoGP.misc.calculate_metrics_fast import calculate_metrics

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

        # Make sure at least 10 species obserservations are present in each subset of the data
        keep_y = (dataset.Y[train_dataset.indices].sum(dim=0) >= split_pct[0] * 10) & (
                dataset.Y[validation_dataset.indices].sum(dim=0) >= split_pct[1] * 10) & (
                         dataset.Y[test_dataset.indices].sum(dim=0) >= split_pct[2] * 10)
        dataset.Y = dataset.Y[:, keep_y]
        dataset.taxon_names = dataset.taxon_names[keep_y]
        dataset.n_species = dataset.Y.shape[1]
        if traits_path:
            dataset.traits = dataset.traits[keep_y, :]
        print(f"Keeping {keep_y.sum().item()} taxons after at least one observation in each split")

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        unique_coordinates = dataset.unique_coords if dataset.using_coordinates else torch.rand(2, 2)

        model = HMSC_GP(unique_coordinates=unique_coordinates, n_latents=n_latents_spatial, n_inducing_points=n_inducing_points_spatial)

        # # Set up the optimizer
        optimizer = Adam({"lr": config["general"]["lr"]})

        # Training
        if config["hmsc"]["cross_validation"]:
            print("CV Depricated")
            # train_svi_cv(
            #     k_fold=config["hmsc"]["k_fold"],
            #     train_dataset=train_dataset,
            #     batch_size=config["general"]["batch_size"],
            #     epoch=config["general"]["n_iter"],
            #     model=model.model,
            #     guide=model.guide,
            #     likelihood=config["hmsc"]["likelihood"],
            #     optimizer=optimizer,
            #     verbose=config["general"]["verbose"]
            # )
        else:
            train_svi(
                train_dataset=train_dataset,
                train_dataloader=train_dataloader,
                epoch=config["general"]["n_iter"],
                model=model.model,
                guide=model.guide,
                likelihood=config["hmsc"]["likelihood"],
                optimizer=optimizer,
                verbose=config["general"]["verbose"]
            )

        # Testing
        test_idx = test_dataset.indices
        # validation_data = dataset.get_batch_data(validation_idx)
        X, Y, coords, traits = test_dataset.dataset.get_batch_data(test_idx)
        training = False

        predictive = Predictive(model.model, guide=model.guide, num_samples=100)

        predict = predictive(X, Y, coords, traits, training, device, config["hmsc"]["likelihood"])["y"].mean(dim=0)

        metrics = calculate_metrics(Y, predict)
        print(metrics)

        res["ROC AUC"].append(metrics["AUC"])
        res["NLL"].append(metrics["NLL"])
        res["MAE"].append(metrics["MAE"])
        res["PR AUC"].append(metrics["PR_AUC"])

    for key, value in res.items():
        print(key, torch.tensor(value).mean(), torch.tensor(value).std())


if __name__ == "__main__":
    import time

    torch.manual_seed(1)

    start = time.time()
    learn_model()
    print(f"Execution time {round(time.time() - start, 2)} seconds")
