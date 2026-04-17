import os.path
import warnings

import torch
import pandas as pd
import polars as pl
import numpy as np


class DataLoad():
    def __init__(self, Y_path, X_path, coords_path, device, normalize_X: bool, traits_path: str="", total_counts_path="", presence_absence_Y=False, verbose=False):
        """
        Initializes the DataLoader with tensor data and batch size.

        Args:
            data (torch.Tensor): The tensor data to be loaded in batches.
            batch_size (int): The number of data points per batch.
        """
        # Device has been set to CPU as it's too large to fit into the GPU
        self.device = device
        self.using_environment = True if X_path else False
        self.using_coordinates = True if coords_path else False
        self.using_traits = True if traits_path else False
        self.using_total_counts = True if total_counts_path else False

        self.verbose = verbose

        self.X = None
        self.Y = None
        self.coords = None
        self.traits = None

        self.unique_coords, self.coords_inverse_indicies = None, None

        self.normalize_X = normalize_X
        self.presence_absence_Y = presence_absence_Y

        self.env_names = None
        self.site_names = None
        self.taxon_names = None

        # To revert back to denormalized
        self.X_continuous = None
        self.X_continuous_mean = None
        self.X_continuous_std = None

        self.coords_mean = None
        self.coords_std = None

        # Data
        self.load_X(X_path)
        self.load_Y(Y_path)
        self.load_coords(coords_path)
        self.load_traits(traits_path)
        self.validate_X()
        self.validate_Y()
        self.transform_X()
        self.transform_Y(total_counts_path)
        self.unique_coordinates()

        self.n_species = self.Y.shape[1]

        self.n_env = self.X.shape[1] if self.using_environment else 0
        self.n_samples = self.Y.shape[0]
        self.n_traits = self.traits.shape[1] if self.using_traits else 0

    def load_X(self, X_path):
        if self.using_environment is False:
            return None

        X = pd.read_csv(X_path, index_col=0)
        self.env_names = np.array(X.columns.tolist())
        self.site_names = np.array(X.index.tolist())
        self.X = torch.tensor(X.values, dtype=torch.float32)

        if self.verbose:
            print(f"Load X: {len(self.site_names)} sample sites and {len(self.env_names)} environmental variables")

    def load_Y(self, Y_path):
        Y = pl.read_csv(Y_path, infer_schema_length=100_000)
        self.site_names = np.array(Y.select(pl.first())[:, 0])
        Y = Y.select(Y.columns[1:])  # Remove species names
        self.taxon_names = np.array(Y.columns)
        self.Y = torch.tensor(Y.to_numpy(), dtype=torch.float32)

        if self.verbose:
            print(f"Load Y: {len(self.site_names)} sample sites and {len(self.taxon_names)} taxons")

    def load_coords(self, coords_path):
        if self.using_coordinates is False:
            return None
        # Distance matrix
        if self.using_coordinates:
            self.coords = torch.tensor(pd.read_csv(coords_path, index_col=0).values, dtype=torch.float32)

            if True:  # Std norm coordinates
                print("Standard Normalizing Coordinates")
                self.coords_mean = self.coords.mean(dim=0)
                self.coords_std = self.coords.std(dim=0)
                self.coords = (self.coords - self.coords_mean) / self.coords_std

    def load_traits(self, traits_path):
        if self.using_traits:
            self.traits = torch.tensor(pd.read_csv(traits_path, index_col=0).values, dtype=torch.float32)

            if self.normalize_X:
                self.traits = (self.traits - self.traits.mean(dim=0)) / self.traits.std(dim=0)

    def validate_X(self):
        if self.using_environment is False:
            return None

        # Check for 0 standard deviation
        env_std = self.X.std(dim=0)
        env_std_non_0 = env_std != 0
        self.env_names = self.env_names[env_std_non_0]
        self.X = self.X[:, env_std_non_0]

        # Check for 25% nan in columns (features)
        keep_features = self.X.isnan().sum(dim=0) < (self.X.shape[0] * 0.25)
        self.env_names = self.env_names[keep_features]
        self.X = self.X[:, keep_features]

        # Check for nan in rows (sites)
        keep_sites = ~torch.any(self.X.isnan(), dim=1)
        self.site_names = self.site_names[keep_sites]
        self.X = self.X[keep_sites, :]
        self.Y = self.Y[keep_sites, :]
        if self.using_coordinates:
            self.coords = self.coords[keep_sites, :]

    def validate_Y(self):
        # Check for 0 standard deviation
        taxon_std = self.Y.std(dim=0)
        taxon_std_non_0 = taxon_std != 0
        self.taxon_names = self.taxon_names[taxon_std_non_0]
        self.Y = self.Y[:, taxon_std_non_0]
        if self.using_traits:
            self.traits = self.traits[taxon_std_non_0, :]

        # TODO: More checks

    def transform_X(self):
        if self.using_environment is False:
            return None

        if self.normalize_X:
            self.X_continuous = ~torch.all((self.X == 0.0) | (self.X == 1.0), dim=0)
            self.X_continuous_mean = self.X[:, self.X_continuous].mean(dim=0)
            self.X_continuous_std = self.X[:, self.X_continuous].std(dim=0)

            self.X[:, self.X_continuous] = (self.X[:, self.X_continuous] - self.X_continuous_mean) / self.X_continuous_std

    def transform_Y(self, total_counts_path):
        if self.presence_absence_Y:
            self.Y = self.Y.bool().float()
        elif total_counts_path:
            # Make sure sum(Y_i)=1
            self.Y = self.Y / self.Y.sum(dim=1, keepdim=True)

            self.total_counts = torch.tensor(pd.read_csv(total_counts_path, index_col=0).values)
            self.Y = (self.Y * self.total_counts).round().int()

    def unique_coordinates(self):
        if self.using_coordinates:
            self.unique_coords, self.coords_inverse_indicies = torch.unique(self.coords, dim=0, return_inverse=True)

            if self.verbose:
                print(f"{len(self.unique_coords)=}")




if __name__ == "__main__":
    Y_path = "../data/clean/Y.csv"
    X_path = "../data/clean/X.csv"
    coords_path = "../data/clean/XY.csv"
    total_counts_path = "../data/clean/total_counts.csv"
    device = torch.device("cpu")
    normalize_X = True
    batch_size = 50

    dataset = DataLoad(Y_path, X_path, coords_path, device, normalize_X, verbose=True, total_counts_path=total_counts_path)

