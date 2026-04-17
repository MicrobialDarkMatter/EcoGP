import torch
import pyro
import pyro.distributions as dist
import numpy as np
import pandas as pd

import os
from configs.config import config

from model import *
# from models.DataSampler import DataSampler

import plotly.graph_objects as go


save_model_path = config["general"]["save_model_path"]

# Loading model and setting learned params
pyro.clear_param_store()
model = torch.load(os.path.join(save_model_path, "model.pt"), weights_only=False)
dataset = torch.load(os.path.join(save_model_path, "dataset.pt"), weights_only=False)
pyro.get_param_store().set_state(torch.load(os.path.join(save_model_path, "param_store.pt"), weights_only=False))

def get_response(n_samples, n_values, variable, iter_range=range(100)):
    predictive = pyro.infer.Predictive(model.model, guide=model.guide, num_samples=n_samples)

    diff_env_inputs = torch.linspace(dataset.X[:, variable].min(), dataset.X[:, variable].max(), n_values)

    means = []
    aboves = []
    belows = []

    #for i in tqdm(range(dataset.X.shape[0])):
    for i in range(100):
    # for i in [1]:
        x = dataset.X[i, :].repeat(n_values, 1)

        x[:, variable] = diff_env_inputs

        range_batch = {'n_samples_batch': n_values, 'n_species': dataset.n_species, 'n_env': dataset.n_env, 'X': x, "training": False}


        samples_z = predictive(range_batch)["z"].squeeze()

        logits_mean = samples_z.mean(dim=0)
        logits_std = samples_z.std(dim=0)

        y_prob_mean = dist.Bernoulli(logits=logits_mean).mean.detach()
        y_prob_above = dist.Bernoulli(logits=logits_mean + 2 * logits_std).mean.detach()
        y_prob_below = dist.Bernoulli(logits=logits_mean - 2 * logits_std).mean.detach()

        means.append(y_prob_mean)
        aboves.append(y_prob_above)
        belows.append(y_prob_below)

    y_prob_mean = torch.stack(means).mean(dim=0)
    y_prob_above = torch.stack(aboves).mean(dim=0)
    y_prob_below = torch.stack(belows).mean(dim=0)

    return y_prob_mean, y_prob_above, y_prob_below


if __name__ == "__main__":
    n_samples = 50  # Number of samples to calculate the individual probabilities from (Higher, to reduce spikes)
    n_values = 250  # Number of points between minimum and maximum for the chosen variable (Higher, the smoother it will be)
    variable_idx = -1  # Index for which variable to look at
    iter_range = [0]  # Sites to include in the calculation of the response curves. Can be multiple as "[0, 1, ...]" or as a "range(10)"

    # Mean predicted probabilities with +-2 std
    mean, above, below = get_response(n_samples, n_values, variable_idx)

    # Converting the features back from standard normalization
    diff_env_inputs = torch.linspace(dataset.X[:, variable_idx].min(), dataset.X[:, variable_idx].max(), n_values)
    x_values = torch.round(diff_env_inputs * dataset.X_continuous_std[variable_idx] + dataset.X_continuous_mean[variable_idx], decimals=2)

    pd.DataFrame(mean.numpy(), columns=dataset.Y_cols_species).to_csv(os.path.join(save_model_path, "mean.csv"), index=False)
    pd.DataFrame(above.numpy(), columns=dataset.Y_cols_species).to_csv(os.path.join(save_model_path, "above.csv"), index=False)
    pd.DataFrame(below.numpy(), columns=dataset.Y_cols_species).to_csv(os.path.join(save_model_path, "below.csv"), index=False)

    pd.DataFrame(x_values.numpy()).to_csv(os.path.join(save_model_path, "x_values.csv"), index=False, header=False)

    # Calculating "importance"
    w = pyro.param("w_loc")
    lengthscale = model.f.covar_module.base_kernel.lengthscale.squeeze()
    outputscale = model.f.covar_module.outputscale
    importance = ((w ** 2 * outputscale ** 2) @ (1 / lengthscale)).detach()

    pd.DataFrame(importance.numpy(), columns=dataset.environmental).to_csv(os.path.join(save_model_path, "importance.csv"), index=False)
