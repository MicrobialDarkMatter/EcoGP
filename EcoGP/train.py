import torch
import pyro
import pyro.distributions as dist
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import tqdm

import pandas as pd

import wandb
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from EcoGP.model import EcoGP
# from EcoGP.model_traits import EcoGP; print("USING TRAITS")
# from EcoGP.model_old import MicroGP as EcoGP; print("USING OLD MODEL!")
# from EcoGP.model_flip import EcoGP; print("USING FLIPPED")
# from EcoGP.model_traits import EcoGP; print("USING TRAITS")

from torch.utils.data import DataLoader, random_split
from EcoGP.DataSampler import DataSampler
from EcoGP.DataLoad import DataLoad
from EcoGP.BetaTraceELBO import BetaTraceELBO

from sklearn import metrics

from likelihoods import DirichletMultinomialLikelihood, BernoulliLikelihood

import importlib
import argparse

class Inputs:
    def __init__(self):
        self.x_path = None
        self.y_path = None
        self.coords_path = None
        self.traits_path = None
        self.total_counts_path = None

        self.n_latents_env = None
        self.n_latents_spatial = None
        self.n_iter = None
        self.n_particles = None
        self.device = None
        self.lr = None
        self.batch_size = None
        self.split_pct = None
        self.n_inducing_points_env = None
        self.n_inducing_points_spatial = None

        self.verbose = None
        self.presence_absence = None
        self.normalize_X = None
        self.likelihood = None
        self.seed = None

        self.save_model_path = None

        self.read_args()

    def read_args(self):
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--config",
            type=str,
            default="config_toy",  # TODO: Change config here or when running in terminal
            help="Name of the config file (without .py extension, must be in configs/)",
        )

        # Settings to override arguments from config for each element
        parser.add_argument("--x_path", type=str, help="Path to environmental covariates (X).")
        parser.add_argument("--y_path", type=str, help="Path to species observations (Y).")
        parser.add_argument("--coords_path", type=str, help="Path to spatial coordinates.")
        parser.add_argument("--traits_path", type=str, help="Path to species traits.")
        parser.add_argument("--total_counts_path", type=str, help="Path to total counts (only count data).")

        parser.add_argument("--n_latents_env", type=int, help="Number of environmental latent GPs.")
        parser.add_argument("--n_latents_spatial", type=int, help="Number of spatial latent GPs.")
        parser.add_argument("--n_iter", type=int, help="Number of training iterations.")
        parser.add_argument("--n_particles", type=int, help="Number of particles for SVI.")
        parser.add_argument("--device", type=str, help="Device to use for training (e.g., 'cpu' or 'cuda').")
        parser.add_argument("--lr", type=float, help="Learning rate for the optimizer.")
        parser.add_argument("--batch_size", type=int, help="Batch size for training.")
        parser.add_argument("--split_pct", type=float, nargs=3, help="Train/validation/test split percentages.")
        parser.add_argument("--n_inducing_points_env", type=int, help="Number of inducing points for environmental GPs.")
        parser.add_argument("--n_inducing_points_spatial", type=int, help="Number of inducing points for spatial GPs.")

        parser.add_argument("--verbose", type=bool, help="Enable verbose output.")
        parser.add_argument("--presence_absence", type=bool, help="Convert relative to presence-absence data.")
        parser.add_argument("--normalize_X", type=bool, help="Normalize features.")
        parser.add_argument("--likelihood", type=str, help="Likelihood model to use.")
        parser.add_argument("--seed", type=int, help="Random seed for reproducibility.")
        parser.add_argument("--save_model_path", type=str, help="Path to save the trained model.")

        args = parser.parse_args()

        # Reads config file
        self.read_config(config_file=args.config)

        # Overrides config if other arguments are provided
        self.owerwrite_config(args=args)

    def read_config(self, config_file: str):
        print(f"Config File: {config_file}")

        config_module = importlib.import_module(f"configs.{config_file}")
        config = config_module.config  # Import the config module

        self.x_path = config["data"]["X_path"]
        self.y_path = config["data"]["Y_path"]
        self.coords_path = config["data"]["coords_path"]
        self.traits_path = config["data"]["traits_path"]
        self.total_counts_path = config["data"]["total_counts_path"]

        self.n_latents_env = config["environmental"]["n_latents"] if self.x_path is not None else None
        self.n_latents_spatial = config["spatial"]["n_latents"] if self.coords_path is not None else None
        self.n_iter = config["general"]["n_iter"]
        self.n_particles = config["general"]["n_particles"]
        self.device = config["general"]["device"]
        self.lr = config["general"]["lr"]
        self.batch_size = config["general"]["batch_size"]
        self.split_pct = config["general"]["split_pct"]
        self.n_inducing_points_env = config["environmental"]["n_inducing_points"]
        self.n_inducing_points_spatial = config["spatial"]["n_inducing_points"]

        self.verbose = config["general"]["verbose"]
        self.presence_absence = config["data"]["presence_absence"]
        self.normalize_X = config["data"]["normalize_X"]
        self.likelihood = config["general"]["likelihood"]
        self.seed = config["general"]["seed"]
        self.save_model_path = config["general"]["save_model_path"]

        # TODO: Verify presence_absence, likelihood, and total_counts_path match. Maybe Y too.

    def owerwrite_config(self, args):
        for attr in vars(self):
            if hasattr(args, attr):
                arg_value = getattr(args, attr)
                if arg_value is None:
                    continue

                if attr == "likelihood":
                    if arg_value == "Bernoulli":
                        arg_value = BernoulliLikelihood
                    elif arg_value == "Dirichlet":
                        arg_value = DirichletMultinomialLikelihood
                    else:
                        raise ValueError(f"Unknown likelihood: {arg_value}")

                setattr(self, attr, arg_value)


def train(inputs: Inputs):
    torch.manual_seed(inputs.seed)

    data = DataLoad(
        Y_path=inputs.y_path,
        X_path=inputs.x_path,
        coords_path=inputs.coords_path,
        traits_path=inputs.traits_path,
        device=inputs.device,
        normalize_X=inputs.normalize_X,
        total_counts_path=inputs.total_counts_path,
        presence_absence_Y=inputs.presence_absence,
        verbose=inputs.verbose
    )

    dataset = DataSampler(data)

    if False:#inputs.coords_path:
        train_indices, validation_indices, test_indices = random_split(torch.arange(dataset.unique_coords.shape[0]),
                                                                       inputs.split_pct,
                                                                       generator=torch.Generator().manual_seed(inputs.seed))

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
        train_dataset, validation_dataset, test_dataset = random_split(dataset, inputs.split_pct,
                                                                       generator=torch.Generator().manual_seed(inputs.seed))

    # Make sure at least 1 species obserservations are present all splits
    # Can't make predictions for a species not present in training
    keep_y = (dataset.Y[train_dataset.indices].sum(dim=0) >= inputs.split_pct[0] * 10) & (
            dataset.Y[validation_dataset.indices].sum(dim=0) >= inputs.split_pct[1] * 10) & (
                     dataset.Y[test_dataset.indices].sum(dim=0) >= inputs.split_pct[2] * 10)
    dataset.Y = dataset.Y[:, keep_y]
    if dataset.using_total_counts:
        dataset.total_counts = (
                    (dataset.Y / dataset.total_counts).sum(dim=1) * dataset.total_counts.squeeze()).int().reshape(-1, 1)
    dataset.taxon_names = dataset.taxon_names[list(keep_y)]
    dataset.n_species = dataset.Y.shape[1]
    if inputs.traits_path:
        dataset.traits = dataset.traits[keep_y, :]
    if inputs.verbose:
        print(f"Keeping {keep_y.sum().item()} taxons with at least {inputs.split_pct} * 10 "
              f"observations per split, respectively.")

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=inputs.batch_size, shuffle=True)

    n_tasks = dataset.n_species
    n_variables = dataset.n_env
    # n_traits = dataset.n_traits
    unique_coordinates = dataset.coords if inputs.coords_path else None
    inputs.n_latents_spatial = inputs.n_latents_spatial if inputs.coords_path else None

    model = EcoGP(
        inputs.n_latents_env,
        n_variables,
        inputs.n_inducing_points_env,
        inputs.n_latents_spatial,
        inputs.n_inducing_points_spatial,
        unique_coordinates,
        likelihood=inputs.likelihood
    ).to(inputs.device)

    optimizer = pyro.optim.Adam({"lr": inputs.lr})
    # elbo = pyro.infer.Trace_ELBO(num_particles=n_particles, vectorize_particles=True, retain_graph=True)

    elbo = BetaTraceELBO(beta=.5, num_particles=inputs.n_particles, vectorize_particles=True, retain_graph=True)

    svi = pyro.infer.SVI(model.model, model.guide, optimizer, elbo)

    model.train()

    lr = inputs.lr

    losses = []
    training = True
    iterator = tqdm.tqdm(range(inputs.n_iter))
    for i in iterator:
        # if i % 2500 == 0 and i > 0:
        #     lr = lr * 0.01
        #     # print(f"New optimizer lr: {lr} at step {i}")
        #     optimizer = pyro.optim.Adam({"lr": lr})
        #     svi = pyro.infer.SVI(model.model, model.guide, optimizer, elbo)
        loss = 0
        for idx in train_dataloader:
            X, Y, coords, traits = train_dataset.dataset.get_batch_data(idx)
            loss += svi.step(X, Y, coords, traits) / Y.nelement()

        iterator.set_postfix(loss=loss)
        losses.append(loss)

    plt.plot(list(range(inputs.n_iter)), losses)
    plt.show()

    # Save model
    if inputs.save_model_path:
        torch.save(model, os.path.join(inputs.save_model_path, "model.pt"))
        pyro.get_param_store().save(os.path.join(inputs.save_model_path, "param_store.pt"))
        torch.save(dataset, os.path.join(inputs.save_model_path, "dataset.pt"))

        # # Save config
        # import pprint
        #
        # with open(os.path.join(save_model_path, 'config.txt'), 'w') as f:
        #     # Create a PrettyPrinter object that writes to the file
        #     pp = pprint.PrettyPrinter(stream=f)
        #     pp.pprint(config)

        # Save parameters
        if inputs.x_path:
            f_mean = model.f.pyro_guide(dataset.X[train_dataset.indices], name_prefix="f_GP").mean.detach().cpu().numpy()
            pd.DataFrame(f_mean, index=dataset.site_names[train_dataset.indices]).to_csv(os.path.join(inputs.save_model_path, "environmental_latents_f.csv"))

            if inputs.traits_path:
                w_loc = (traits @ pyro.param("gamma_loc").T).T.detach().cpu().numpy()
                pd.DataFrame(w_loc, columns=dataset.taxon_names).to_csv(os.path.join(inputs.save_model_path, "weights_traits_env_w.csv"))

                gamma_loc = pyro.param("gamma_loc").detach().cpu().numpy()
                #pd.DataFrame(gamma_loc, index=[f"Trait_{i}" for i in range(gamma_loc.shape[0])], columns=[f"Env_Latent_{i}" for i in range(gamma_loc.shape[1])]).to_csv(os.path.join(inputs.save_model_path, "gamma_loc.csv"))
            else:
                w_loc = pyro.param("w_loc").detach().cpu().numpy()
                pd.DataFrame(w_loc, columns=dataset.taxon_names).to_csv(os.path.join(inputs.save_model_path, "weights_environmental_w.csv"))

            if inputs.coords_path:
                g_mean = model.g.pyro_guide(dataset.coords[train_dataset.indices],
                                            name_prefix="g_GP").mean.detach().cpu().numpy()
                pd.DataFrame(g_mean, index=dataset.site_names[train_dataset.indices]).to_csv(
                    os.path.join(inputs.save_model_path, "spatial_latents_g.csv"))

                v_loc = pyro.param("v_loc").detach().cpu().numpy()
                pd.DataFrame(v_loc, columns=dataset.taxon_names).to_csv(os.path.join(inputs.save_model_path, "weights_spatial_v.csv"))

            bias_loc = pyro.param("bias_loc").detach().cpu().numpy()
            pd.DataFrame(bias_loc, index=dataset.taxon_names).to_csv(os.path.join(inputs.save_model_path, "bias_loc.csv"))

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    prob_list = []
    y_train_list = []
    for idx in train_dataloader:
        X, Y, coords, traits = train_dataset.dataset.get_batch_data(idx)
        res = model.forward(X, Y, coords, traits).detach()

        prob_list.append(res)
        y_train_list.append(Y / (dataset.total_counts[idx] if dataset.using_total_counts else 1))

    prob = torch.concat(prob_list)
    train_Y = torch.concat(y_train_list)
    del prob_list, y_train_list

    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(train_Y, prob)

    # J = tpr - fpr
    # best_idx = J.argmax()
    # best_threshold = thresholds[best_idx]

    points = torch.stack([torch.tensor(fpr), torch.tensor(tpr)], dim=1)  # (N, 2)
    dists = torch.norm(points - torch.tensor([0.0, 1.0]), dim=1)  # Euclidean distance
    best_idx = torch.argmin(dists)
    best_threshold = thresholds[best_idx]
    print("train, threshold: ", best_threshold)

    cm = confusion_matrix(train_Y, (prob >= best_threshold).float(),
                          # labels=clf.classes_
                          )
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  # display_labels=clf.classes_
                                  )
    disp.plot()
    plt.show()

    plt.figure()
    plt.plot(fpr, tpr, label=f"TRAIN // ROC curve (AUC = {1337:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

    TN, FP, FN, TP = cm.flatten()
    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)
    precision = TP / (TP + FP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print(f"TRAIN: precision: {precision:.3f}, recall: {recall:.3f}, specificity: {specificity:.3f}, accuracy: {accuracy:.3f}")

    # Validation
    validation_dataloader = DataLoader(dataset=validation_dataset,
                                 batch_size=inputs.batch_size,
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

    torch.save(prob, os.path.join(inputs.save_model_path, "Y_pred_validation.pt"))
    torch.save(validation_Y, os.path.join(inputs.save_model_path, "Y_true_validation.pt"))

    from EcoGP.misc.calculate_metrics_fast import calculate_metrics

    metrics = calculate_metrics(validation_Y, prob)
    print("\nValidation", metrics)

    cm = confusion_matrix(validation_Y, (prob >= best_threshold).float(),
                          # labels=clf.classes_
                          )
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  # display_labels=clf.classes_
                                  )
    disp.plot()
    plt.show()

    fpr, tpr, thresholds = roc_curve(validation_Y, prob)

    J = tpr - fpr
    best_idx = J.argmax()
    print("vali, threshold: ", thresholds[best_idx])

    plt.figure()
    plt.plot(fpr, tpr, label=f"VALIDATION // ROC curve (AUC = {metrics['AUC']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

    TN, FP, FN, TP = cm.flatten()
    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)
    precision = TP / (TP + FP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print(
        f"VALIDATION: precision: {precision:.3f}, recall: {recall:.3f}, specificity: {specificity:.3f}, accuracy: {accuracy:.3f}")

    # test
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=inputs.batch_size,
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

    torch.save(prob, os.path.join(inputs.save_model_path, "Y_pred_test.pt"))
    torch.save(test_Y, os.path.join(inputs.save_model_path, "Y_true_test.pt"))

    from EcoGP.misc.calculate_metrics_fast import calculate_metrics

    metrics = calculate_metrics(test_Y, prob)
    print("\nTest", metrics)

    cm = confusion_matrix(test_Y, (prob >= best_threshold).float())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    fpr, tpr, thresholds = roc_curve(test_Y, prob)

    J = tpr - fpr
    best_idx = J.argmax()
    print("test, threshold: ", thresholds[best_idx])

    plt.figure()
    plt.plot(fpr, tpr, label=f"TEST // ROC curve (AUC = {metrics['AUC']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

    TN, FP, FN, TP = cm.flatten()
    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)
    precision = TP / (TP + FP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print(
        f"TEST: precision: {precision:.3f}, recall: {recall:.3f}, specificity: {specificity:.3f}, accuracy: {accuracy:.3f}")

    if X is not None:
        f_lengthscales, f_lengthscales_idx = model.f.covar_module.base_kernel.lengthscale.squeeze().sort()
        k = 5
        print(f"Top {k} Environmental Length Scales: {f_lengthscales[:k]}, "
              f"Variables: {dataset.env_names[f_lengthscales_idx[:k]]}")
    if coords is not None:
        g_lengthscales, g_lengthscales_idx = model.g.covar_module.base_kernel.lengthscale.squeeze().sort()
        k = 5
        print(f"Top {k} Spatial Length Scales: {g_lengthscales[:k]}, "
              f"Variables idx: {g_lengthscales_idx[:k]}")



    print("DONE")



if __name__ == "__main__":
    inputs = Inputs()
    train(inputs)
