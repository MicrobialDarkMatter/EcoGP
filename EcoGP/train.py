import torch
import pyro
import pyro.distributions as dist
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import tqdm

import wandb
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from EcoGP.model import EcoGP
# from EcoGP.model_old import MicroGP as EcoGP; print("USING OLD MODEL!")
# from EcoGP.model_flip import EcoGP; print("USING FLIPPED")
# from EcoGP.model_traits import EcoGP; print("USING TRAITS")

from torch.utils.data import DataLoader, random_split
from EcoGP.DataSampler import DataSampler
from EcoGP.DataLoad import DataLoad
from EcoGP.BetaTraceELBO import BetaTraceELBO

from sklearn import metrics

if __name__ == "__main__":
    import importlib
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config_clean_unique",  # TODO: Change config here or when running in terminal
        help="Name of the config file (without .py extension, must be in configs/)",
    )

    # To override arguments from config
    parser.add_argument('--n_latents_env', type=int)
    parser.add_argument('--n_inducing_points_env', type=int)
    parser.add_argument('--n_latents_spatial', type=int)
    parser.add_argument('--n_inducing_points_spatial', type=int)
    parser.add_argument('--save_model_path', type=str)
    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    print(f"Config File: {args.config}")

    config_module = importlib.import_module(f"configs.{args.config}")
    config = config_module.config  # Import the config module

    # Overrides config
    if args.n_latents_env:
        config["environmental"]["n_latents"] = args.n_latents_env
    if args.n_inducing_points_env:
        config["environmental"]["n_inducing_points"] = args.n_inducing_points_env
    if args.n_latents_spatial:
        config["spatial"]["n_latents"] = args.n_latents_spatial
    if args.n_inducing_points_spatial:
        config["spatial"]["n_inducing_points"] = args.n_inducing_points_spatial
    if args.save_model_path:
        config["general"]["save_model_path"] = args.save_model_path
    if args.seed is not None:
        config["general"]["seed"] = args.seed

    # ARGUMENTS
    # environment = config["additive"]["environment"]
    # spatial = config["additive"]["spatial"]
    # traits = config["additive"]["traits"]

    x_path = config["data"]["X_path"]
    y_path = config["data"]["Y_path"]
    coords_path = config["data"]["coords_path"]
    traits_path = config["data"]["traits_path"]
    total_counts_path = config["data"]["total_counts_path"]
    #hierarchy_path = config["data"]["hierarchy_path"]

    n_latents_env = config["environmental"]["n_latents"] if x_path is not None else None
    n_latents_spatial = config["spatial"]["n_latents"] if coords_path is not None else None
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

    save_model_path = config["general"]["save_model_path"]
    # STOP ARGUMENTS

    torch.manual_seed(seed)

    data = DataLoad(
        Y_path=y_path,
        X_path=x_path,
        coords_path=coords_path,
        traits_path=traits_path,
        device=device,
        normalize_X=normalize_X,
        total_counts_path=total_counts_path,
        presence_absence_Y=presence_absence,
        verbose=verbose
    )

    dataset = DataSampler(data)

    if coords_path:
        train_indices, validation_indices, test_indices = random_split(torch.arange(dataset.unique_coords.shape[0]),
                                                                       split_pct,
                                                                       generator=torch.Generator().manual_seed(seed))

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
                                                                       generator=torch.Generator().manual_seed(seed))

    # Make sure at least 1 species obserservations are present all splits
    # Can't make predictions for a species not present in training
    keep_y = (dataset.Y[train_dataset.indices].sum(dim=0) >= split_pct[0] * 10) & (
                dataset.Y[validation_dataset.indices].sum(dim=0) >= split_pct[1] * 10) & (
                dataset.Y[test_dataset.indices].sum(dim=0) >= split_pct[2] * 10)
    dataset.Y = dataset.Y[:, keep_y]
    if dataset.using_total_counts:
        dataset.total_counts = ((dataset.Y / dataset.total_counts).sum(dim=1) * dataset.total_counts.squeeze()).int().reshape(-1, 1)
    dataset.taxon_names = dataset.taxon_names[keep_y]
    dataset.n_species = dataset.Y.shape[1]
    if traits_path:
        dataset.traits = dataset.traits[keep_y, :]
    if verbose:
        print(f"Keeping {keep_y.sum().item()} taxons with at least {split_pct} * 10 "
              f"observations per split, respectively.")

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    n_tasks = dataset.n_species
    n_variables = dataset.n_env
    # n_traits = dataset.n_traits
    unique_coordinates = dataset.coords if coords_path else None

    model = EcoGP(
        n_latents_env,
        n_variables,
        n_inducing_points_env,
        n_latents_spatial,
        n_inducing_points_spatial,
        unique_coordinates,
        # environment=environment,
        # spatial=spatial,
        # traits=traits,
        likelihood=likelihood
    ).to(device)

    optimizer = pyro.optim.Adam({"lr": lr})
    # elbo = pyro.infer.Trace_ELBO(num_particles=n_particles, vectorize_particles=True, retain_graph=True)

    elbo = BetaTraceELBO(beta=.5, num_particles=n_particles, vectorize_particles=True, retain_graph=True)

    svi = pyro.infer.SVI(model.model, model.guide, optimizer, elbo)

    model.train()

    losses = []
    training = True
    iterator = tqdm.tqdm(range(n_iter))
    for i in iterator:
        loss = 0
        for idx in train_dataloader:
            X, Y, coords, traits = train_dataset.dataset.get_batch_data(idx)
            loss += svi.step(X, Y, coords, traits) / Y.nelement()

        iterator.set_postfix(loss=loss)
        losses.append(loss)

    plt.plot(list(range(n_iter)), losses)
    plt.show()

    # Save model
    if save_model_path:
        torch.save(model, os.path.join(save_model_path, "model.pt"))
        pyro.get_param_store().save(os.path.join(save_model_path, "param_store.pt"))
        # torch.save(dataset, os.path.join(save_model_path, "dataset.pt"))

        # Save config
        import pprint

        with open(os.path.join(save_model_path, 'config.txt'), 'w') as f:
            # Create a PrettyPrinter object that writes to the file
            pp = pprint.PrettyPrinter(stream=f)
            pp.pprint(config)

    # Validation
    validation_dataloader = DataLoader(dataset=validation_dataset,
                                 batch_size=batch_size,
                                 shuffle=True)

    prob_list = []
    y_validation_list = []
    for idx in validation_dataloader:
        X, Y, coords, traits = validation_dataset.dataset.get_batch_data(idx)
        res = model.forward(X, Y, coords, traits).detach()

        prob_list.append(res)
        y_validation_list.append(Y / (dataset.total_counts[idx] if dataset.using_total_counts else 1))

    prob = torch.concat(prob_list)
    validation_Y = torch.concat(y_validation_list)
    del prob_list, y_validation_list

    torch.save(prob, os.path.join(save_model_path, "Y_pred_validation.pt"))
    torch.save(validation_Y, os.path.join(save_model_path, "Y_true_validation.pt"))

    from EcoGP.misc.calculate_metrics_fast import calculate_metrics

    metrics = calculate_metrics(validation_Y, prob)
    print(metrics)

    # test
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=True)

    prob_list = []
    y_test_list = []
    for idx in test_dataloader:
        X, Y, coords, traits = validation_dataset.dataset.get_batch_data(idx)
        res = model.forward(X, Y, coords, traits).detach()

        prob_list.append(res)
        y_test_list.append(Y / (dataset.total_counts[idx] if dataset.using_total_counts else 1))

    prob = torch.concat(prob_list)
    test_Y = torch.concat(y_test_list)
    del prob_list, y_test_list

    torch.save(prob, os.path.join(save_model_path, "Y_pred_test.pt"))
    torch.save(test_Y, os.path.join(save_model_path, "Y_true_test.pt"))

    from EcoGP.misc.calculate_metrics_fast import calculate_metrics

    metrics = calculate_metrics(test_Y, prob)
    print(metrics)

    print("Done")
