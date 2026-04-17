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

import gpytorch

from models.MultitaskVariationalStrategy import MultitaskVariationalStrategy

from models.DataSampler import DataSampler as CustomDataSubsampling  # TODO: Fix import when moved
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


class SpatialGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, num_latents, unique_coordinates):
        self.num_latents = num_latents
        # Let's use a different set of inducing points for each latent function
        num_coords = unique_coordinates.size(0)

        n_inducing = 2000

        #inducing_points = inducing_points[:, range(n_inducing), :]
        inducing_points = unique_coordinates[torch.stack([torch.randperm(num_coords)[:n_inducing] for _ in range(self.num_latents)]), :]
        # inducing_points += torch.randn(*inducing_points.shape) #* 0.001

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        )

        variational_strategy = MultitaskVariationalStrategy(  # CustomVariationalStrategy
            base_variational_strategy=gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=False
            ),
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch, so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            HaversineRBFKernel(# CustomSpatialKernel(#HaversineRBFKernel(#HaversineRBFKernel(#
                lengthscale_prior=gpytorch.priors.NormalPrior(loc=5, scale=5),
                batch_shape=torch.Size([num_latents]),
            ),
            outputscale_prior=gpytorch.priors.GammaPrior(rate=1, concentration=1),
            batch_shape=torch.Size([num_latents])
        )

        self.covar_module.base_kernel.lengthscale = torch.rand(num_latents, 1, 1) * 5
        self.covar_module.outputscale = torch.rand(num_latents, 1, 1)


class HMSC_GP(pyro.nn.PyroModule):
    def __init__(self, unique_coordinates):
        super().__init__()

        self.n_spatial = 5  # TODO: put in init
        self.eta = SpatialGPModel(num_latents=self.n_spatial, unique_coordinates=unique_coordinates)

    def model(self, batch, likelihood):
        pyro.module("model", self)
        device = batch.get("device")

        n_species = batch.get("n_species")
        n_env = batch.get("n_env")
        n_traits = batch.get("n_traits")
        env_plate = pyro.plate(name="env_plate", size=n_env, dim=-2)
        trait_plate = pyro.plate(name="trait_plate", size=n_traits, dim=-2)

        if "traits" in batch:
            with trait_plate:
                gamma = pyro.sample("gamma", dist.Normal(loc=torch.zeros(n_traits, n_env, device=device),
                                                         scale=torch.ones(n_traits, n_env, device=device)))

            beta_loc = (batch.get("traits") @ gamma).T
        else:
            beta_loc = torch.zeros(n_env, n_species, device=device)

        with env_plate:
            # beta = pyro.sample("beta", dist.MultivariateNormal(loc=torch.zeros(n_env, n_species, device=device),
            #                                                    covariance_matrix=torch.tile(V, (n_env, 1, 1))))
            beta = pyro.sample("beta", dist.Normal(loc=beta_loc,
                                                   scale=torch.ones(n_env, n_species, device=device)))

        # Define the latent means (L)
        L = batch.get("X") @ beta

        if "dist" in batch:
            eps = self.hmsc_spatial_model(batch)
            L += eps

        # Changes model likelihood for final step
        self.model_likelihood(L, batch, likelihood)


    def hmsc_spatial_model(self, batch):
        device = batch.get("device")

        lengthscale = torch.ones(self.n_spatial, device=device)  # TODO: Add such that it can be modified!

        n_latents = self.n_spatial

        # ### ETAS ### #
        inverse_indices = batch.get("batch_inverse")
        n_locs_batch = batch.get("n_locs_batch")

        distance_matrix = batch.get("dist").to(device)

        if n_locs_batch < 10:
            warnings.warn(f'Less than 10 unique sites')

        eta_lengthscale = pyro.param("eta_lengthscale", lengthscale, constraint=constraints.positive)

        eta_covariance = get_eta_covariance_matrix(distance_matrix, eta_lengthscale)

        if n_latents == 1:
            print("Etas not working with squeeze in the end for n_latents == 1")
        with pyro.plate("etas_plate", n_latents, dim=-1):
            etas = pyro.sample(f"etas", dist.MultivariateNormal(loc=torch.zeros(n_latents, n_locs_batch, device=device),
                                                                covariance_matrix=eta_covariance).to_event(0)).squeeze()

        etas = etas.T[inverse_indices]

        # ### LAMBDAS ### #
        n_species = batch.get("n_species")
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


    def guide(self, batch, likelihood):
        device = batch.get("device")

        n_species = batch.get("n_species")
        n_env = batch.get("n_env")
        n_traits = batch.get("n_traits")

        # species_plate = pyro.plate("species_plate", n_species, dim=-1)
        env_plate = pyro.plate("env_plate", n_env, dim=-2)
        trait_plate = pyro.plate(name="trait_plate", size=n_traits, dim=-2)

        # with species_plate:
        #     sigma = pyro.param("sigma", torch.ones(n_species, device=device), constraint=constraints.positive)

        if "traits" in batch:
            gamma_loc = pyro.param("gamma_loc", torch.zeros(n_traits, n_env, device=device))
            gamma_scale = pyro.param("gamma_scale", torch.ones(n_traits, n_env, device=device),
                                     constraint=constraints.positive)

            with trait_plate:
                gamma = pyro.sample("gamma", dist.Normal(loc=gamma_loc, scale=gamma_scale))

        beta_loc = pyro.param("beta_loc", torch.zeros(n_env, n_species, device=device))
        beta_scale = pyro.param("beta_scale", torch.ones(n_env, n_species, device=device), constraint=constraints.positive)

        with env_plate:
            beta = pyro.sample("beta", dist.Normal(loc=beta_loc, scale=beta_scale))

        if "dist" in batch:
            n_locs_total = batch.get("n_locs_total")
            # eta
            n_latents = self.n_spatial

            unique_batch_sites = batch.get("unique_batch_locs")  # _ = reverse unique, but not neded in guide?

            with pyro.plate("etas_plate", n_latents, dim=-1):  # TODO subsample=unique_batch_sites not working
                eta_loc = pyro.param("eta_loc", torch.zeros(n_latents, n_locs_total, device=device))
                eta_scale = pyro.param("eta_scale", torch.ones(n_latents, n_locs_total, device=device),
                                       constraint=constraints.positive)

                # Swapped locs and latents to get the right shape (reshape back after sampling)
                etas = pyro.sample(f"etas", dist.Normal(loc=eta_loc[:, unique_batch_sites],
                                                        scale=eta_scale[:, unique_batch_sites]).to_event(1))

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

        self.guide_likelihood(batch, likelihood)


    def model_likelihood(self, L, batch, likelihood):
        training = batch.get("training", True)
        device = batch.get("device")

        n_samples = batch.get("n_samples_batch")
        n_species = batch.get("n_species")

        samples_plate = pyro.plate(name="samples_plate", size=n_samples, dim=-2)
        species_plate = pyro.plate(name="species_plate", size=n_species, dim=-1)

        match likelihood:
            case "normal":
                with species_plate:
                    sigma = pyro.param("sigma", torch.ones(n_species, device=device), constraint=constraints.positive)

                # Define the likelihood of the observed data
                with samples_plate, species_plate:
                    pyro.sample(f"y", dist.Normal(L, sigma), obs=batch.get("Y") if training else None)

            case "bernoulli":
                with samples_plate, species_plate:
                    pyro.sample(f"y", dist.Bernoulli(logits=L), obs=batch.get("Y").bool().float() if training else None)

            case "dirichlet_multinomial":
                L = torch.nn.Softplus()(L)

                L += 1e-8  # Added due to numerical error when 0.0

                with samples_plate:
                    pyro.sample(f"y", dist.DirichletMultinomial(L, total_count=batch.get("Y").sum(dim=1), is_sparse=True),
                                obs=batch.get("Y") if training else None)
            case _:
                warnings.warn("Likelihood not defined!")


    def guide_likelihood(self, batch, likelihood):
        device = batch.get("device")

        match likelihood:
            case "normal":
                n_species = batch.get("n_species")
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
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    # Store the loss values
    loss_list = []

    # Perform inference
    for step in range(epoch):
        # print(f"Starting step {step} of {epoch}")
        loss = 0
        for idx in train_dataloader:
            batch = train_dataset.dataset.get_batch_data(idx)
            loss += svi.step(batch, likelihood)
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


def train_svi_cv(k_fold, train_dataset, batch_size, epoch, model, guide, likelihood, optimizer, verbose):
    kf = KFold(n_splits=k_fold, shuffle=True)

    # Loop through each fold
    for fold, (train_idx, validation_idx) in enumerate(kf.split(train_dataset)):
        print(f"\n~~~~~ Fold {fold + 1} ~~~~~")

        # Define the data loaders for the current fold
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(train_idx),
        )

        train_svi(
            train_dataset=train_dataset,
            train_dataloader=train_dataloader,
            epoch=epoch,
            model=model,
            guide=guide,
            likelihood=likelihood,
            optimizer=optimizer,
            verbose=verbose,
        )

        # Evaluate
        validation_idx = train_dataset.indices
        # test_data = dataset.get_batch_data(test_idx)
        validation_data = train_dataset.dataset.get_batch_data(validation_idx)
        validation_data["training"] = False

        predictive = Predictive(model, guide=guide, num_samples=50)

        predict = predictive(validation_data, likelihood)["y"].mean(dim=0)

        auc_per_species = [
            metrics.roc_auc_score(validation_data.get("Y")[:, i].bool().int(), predict[:, i]) if not all(
                validation_data.get("Y")[:, i] == 0) else float("nan") for i in
            range(validation_data.get("Y").shape[1])
        ]

        auc = torch.tensor(auc_per_species)
        means_tensor = auc[~torch.isnan(auc)]

        if True:  # Metrics
            above_average = (means_tensor > 0.5).sum().item()
            below_average = (means_tensor <= 0.5).sum().item()

            pct_good = above_average / (above_average + below_average)
            print(f"Species ROC above 50%: {pct_good * 100:.2f}%")

            print(f"Species ROC above 50% * average: {pct_good * means_tensor.mean():.4f}")


def learn_model():
    from configs.config import config

    dataset = CustomDataSubsampling(
        Y_path=config["data"]["Y_path"],
        X_path=config["data"]["X_path"],
        coords_path=config["data"]["coords_path"],
        traits_path=config["data"]["traits_path"],
        device=config["general"]["device"],
        normalize_X=config["data"]["normalize_X"],
        prevalence_threshold=config["data"]["prevalence_threshold"]
    )

    unique_coordinates = dataset.unique_coords if dataset.using_coordinates else torch.rand(2, 2)

    model = HMSC_GP(unique_coordinates=unique_coordinates)

    train_size = int(config["general"]["train_pct"] * len(dataset))
    test_size = len(dataset) - train_size
    train_size = train_size - test_size
    val_size = test_size

    train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size],
                                                            generator=torch.Generator().manual_seed(42))
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config["general"]["batch_size"], shuffle=True)

    # Make sure at least 10 species obserservations are present in each subset of the data
    keep_y = (dataset.Y[train_dataset.indices].sum(dim=0) >= 10) & (
            dataset.Y[test_dataset.indices].sum(dim=0) >= 10)
    dataset.Y = dataset.Y[:, keep_y]
    dataset.traits = dataset.traits[keep_y, :]
    dataset.n_species = dataset.Y.shape[1]

    # # Set up the optimizer
    optimizer = Adam({"lr": config["general"]["lr"]})

    # Training
    if config["hmsc"]["cross_validation"]:
        train_svi_cv(
            k_fold=config["hmsc"]["k_fold"],
            train_dataset=train_dataset,
            batch_size=config["general"]["batch_size"],
            epoch=config["general"]["n_iter"],
            model=model.model,
            guide=model.guide,
            likelihood=config["hmsc"]["likelihood"],
            optimizer=optimizer,
            verbose=config["general"]["verbose"]
        )
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
    # test_data = dataset.get_batch_data(test_idx)
    test_data = test_dataset.dataset.get_batch_data(test_idx)
    test_data["training"] = False

    predictive = Predictive(model.model, guide=model.guide, num_samples=100)

    predict = predictive(test_data, config["hmsc"]["likelihood"])["y"].mean(dim=0)

    auc_per_species = [
        metrics.roc_auc_score(test_data.get("Y")[:, i].bool().int(), predict[:, i]) if not all(
            test_data.get("Y")[:, i] == 0) else float("nan") for i in
        range(test_data.get("Y").shape[1])
    ]

    auc = torch.tensor(auc_per_species)
    means_tensor = auc[~torch.isnan(auc)]

    from models.misc.calculate_metrics import calculate_metrics
    res = calculate_metrics(test_data.get("Y"), predict)
    print(res)


if __name__ == "__main__":
    import time

    torch.manual_seed(1)

    start = time.time()
    learn_model()
    print(f"Execution time {round(time.time() - start, 2)} seconds")
