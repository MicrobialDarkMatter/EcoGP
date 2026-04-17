import warnings

import torch
import pandas as pd
import polars as pl
import numpy as np

from torch.utils.data import DataLoader, Dataset

from models.misc.distance_matrix import get_distance_matrix  # TODO: Fix import when moved


class DataSampler(Dataset):
    def __init__(self, Y_path, X_path, coords_path, device, normalize_X: bool, traits_path: str="", prevalence_threshold=0):
        """
        Initializes the DataLoader with tensor data and batch size.

        Args:
            data (torch.Tensor): The tensor data to be loaded in batches.
            batch_size (int): The number of data points per batch.
        """
        # Device has been set to CPU as it's too large to fit into the GPU
        self.device = device
        self.using_coordinates = True if coords_path else False
        self.using_traits = True if traits_path else False

        # self.Y = torch.tensor(pd.read_csv(Y_path, index_col=0).values, dtype=torch.float32).to(device)
        Y = pl.read_csv(Y_path)
        Y_idx_sites = Y.select(pl.first())
        Y = Y.select(Y.columns[1:])  # Remove species names
        Y_cols_species = np.array(Y.columns)
        self.Y = torch.tensor(Y.to_numpy(), dtype=torch.float32)

        self.Y_idx_sites = Y_idx_sites
        self.Y_cols_species = Y_cols_species

        XData = pd.read_csv(X_path, index_col=0)

        XData = XData.loc[:, XData.std() != 0]  # Remove variables with no variance

        # Removing spatial data
        XData = XData.loc[:, ~XData.columns.str.contains("georeg", case=False)]
        XData = XData.loc[:, ~XData.columns.str.contains("ogc_lite_pi0000", case=False)]
        # Add lat,lon
        # lon_lat = pd.read_csv(coords_path, index_col=0)
        # XData[["longitude", "latitude"]] = lon_lat.values

        self.sites = list(XData.index)
        self.environmental = list(XData.columns)
        self.X = torch.tensor(XData.values, dtype=torch.float32)

        # self.X += torch.rand_like(self.X) * 1e-1  # TODO: Fix adding of noice to make cov positive semidefinite

        # Normalize X
        if normalize_X:
            self.X_continuous = ~torch.all((self.X == 0.0) | (self.X == 1.0), dim=0)
            self.X_continuous_mean = self.X[:, self.X_continuous].mean(dim=0)
            self.X_continuous_std = self.X[:, self.X_continuous].std(dim=0)
            if any(self.X_continuous_std == 0):
                warnings.warn("Zero devision in std")
                self.X_continuous_std[self.X_continuous_std == 0] = 1.0
            self.X[:, self.X_continuous] = (self.X[:,
                                            self.X_continuous] - self.X_continuous_mean) / self.X_continuous_std

            # self.X = self.X[:, self.X_continuous]  # TODO: Remove after testing

            # Intercept
            self.X = torch.cat((torch.ones(self.X.shape[0], 1), self.X), dim=1)
            self.environmental = ["intercept"] + self.environmental
            # self.X_og_mean = torch.cat((torch.tensor([0]), self.X_og_mean))
            # self.X_og_std = torch.cat((torch.tensor([1]), self.X_og_std))

        # Normalize Y
        self.Y = self.Y.bool().float()

        # All counts are zero in column 5, so we first drop it
        # keep_Y = self.Y.bool().sum(dim=0) > 0
        all_sites, all_species = self.Y.shape
        keep_Y = self.Y.bool().sum(dim=0) > all_sites * prevalence_threshold  # Only keeping species with % abundance or more
        keep_Y = keep_Y & self.Y.std(
            dim=0).bool()  # TODO: remove? Not keeping species that are either all present or absent
        self.Y = self.Y[:, keep_Y]
        self.Y_cols_species = self.Y_cols_species[keep_Y]
        print(f"Keeping {len(self.Y_cols_species)} species ({len(self.Y_cols_species) / all_species * 100:.2f}%)")

        # Distance matrix
        if self.using_coordinates:
            self.coords = torch.tensor(pd.read_csv(coords_path, index_col=0).values, dtype=torch.float32)

            if False:  # to northing/easting km
                print("CHANGING TO NORTHING EASTING COORDINATES")
                import pyproj

                p = pyproj.Proj(proj='utm', zone=32, ellps='WGS84')
                northing, easting = p(self.coords[:, 0].numpy(), self.coords[:, 1].numpy())
                northing, easting = torch.from_numpy(northing), torch.from_numpy(easting)
                self.coords = torch.stack([northing, easting], dim=1)
                # self.coords = torch.round(self.coords, decimals=-1)

                self.coords = self.coords / 1e3
                self.coords = self.coords.int().float()

            if True:  # Std norm coordinates
                print("Standard Normalizing Coordinates")
                self.coords = (self.coords - self.coords.mean(dim=0)) / self.coords.std(dim=0)

            self.unique_coords, self.coords_inverse_indicies = torch.unique(self.coords, dim=0, return_inverse=True)
            print(f"{len(self.unique_coords)=}")
            # self.dist_matrix = get_distance_matrix(self.unique_coords)

            self.n_locs = len(self.unique_coords)

        # Traits
        if self.using_traits:
            self.traits = torch.tensor(pd.read_csv(traits_path, index_col=0).values, dtype=torch.float32)
            self.traits = self.traits[keep_Y, :]

            if normalize_X:
                self.traits = (self.traits - self.traits.mean(dim=0)) / self.traits.std(dim=0)

                # # Intercept
                # self.traits = torch.cat((torch.ones(self.traits.shape[0], 1), self.traits), dim=1)

        # TODO: FIX THIS HARD CODING
        # self.X = self.X[:, -10:]
        # self.X = self.X[:500, -20:]
        # self.Y = self.Y[:500, :200]#[0, 1, 2, 3, 4, 755, 1770, 3090, 3240, 3669]]
        # if self.using_traits:
        #     self.traits = self.traits[:200, :]

        if True:  # Distances in X
            self.X_dist_matrix = torch.cdist(self.X, self.X)

        if self.using_traits and True:  # Distances in X
            self.traits_dist_matrix = torch.cdist(self.traits, self.traits)

        self.n_species = self.Y.shape[1]
        self.n_env = self.X.shape[1]
        assert self.Y.shape[0] == self.X.shape[0], "Samples in X and Y differ!"
        self.n_samples = self.Y.shape[0]
        if self.using_traits:
            self.n_traits = self.traits.shape[1]

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return idx  # , dist_idx

    def get_batch_data(self, idx):
        batch = {
            # Metadata
            "device": self.device,
            "n_samples_total": self.n_samples,
            "n_samples_batch": len(idx),
            "n_species": self.n_species,
            "n_env": self.n_env,
            # Data
            "X": self.get_X(idx).to(self.device),
            "Y": self.get_Y(idx).to(self.device),
        }
        if self.using_coordinates:
            unique_locs_idx, reverse = self.get_dist_idx_reverse(idx)
            # Data
            batch.update({
                #"dist": self.get_dist(unique_locs_idx).to(self.device),
                "coords": self.coords[unique_locs_idx].to(self.device)
            })
            # Metadata
            batch.update({
                "n_locs_total": self.n_locs,
                "n_locs_batch": batch.get("coords").shape[0],
                "unique_batch_locs": unique_locs_idx,
                "batch_inverse": reverse,
            })
        if True:  # X distances
            # Data
            batch.update({"X_dist": self.get_X_dist(idx).to(self.device)})

        if True and self.using_traits:  # Trait distances TODO
            batch.update({"traits_dist": self.traits_dist_matrix.to(self.device)})

        if self.using_traits:
            batch.update({"traits": self.traits.to(self.device)})
            batch.update({
                "n_traits": batch.get("traits").shape[1],
            })

        return batch

    def get_dist_batch(self, idx):
        return self.coords_inverse_indicies[idx]

    def get_dist_idx_reverse(self, idx):
        batch = self.get_dist_batch(idx)
        unique, reverse = torch.unique(batch, dim=0, return_inverse=True)
        return unique, reverse

    def get_X(self, idx=None):
        if idx is None:
            return self.X
        else:
            return self.X[idx].to(self.device)

    def get_Y(self, idx=None):
        if idx is None:
            return self.Y
        else:
            return self.Y[idx].to(self.device)

    # def get_dist(self, idx=None):
    #     """
    #
    #     :param idx: Be the unique idx from get_dis_idx_reverse
    #     :return:
    #     """
    #     assert self.using_coordinates, "No coordinates given!"
    #     if idx is None:
    #         return self.dist_matrix
    #     else:
    #         return self.dist_matrix[idx][:, idx].to(self.device)

    def get_X_dist(self, idx=None):
        """

        :param idx: Be the unique idx from get_dis_idx_reverse
        :return:
        """
        if idx is None:
            return self.X_dist_matrix
        else:
            return self.X_dist_matrix[idx][:, idx].to(self.device)

    def get_Y_mean_std(self):
        return self.Y_mean, self.Y_std

    def get_Y_idx_cols(self):
        return self.Y_idx_sites, self.Y_cols_species


if __name__ == "__main__":
    Y_path = "../data/full/Y.csv"
    X_path = "../data/full/X.csv"
    coords_path = "../data/full/XY.csv"
    device = torch.device("cpu")
    normalize_X = True
    normalize_Y = True
    batch_size = 50

    dataset = DataSampler(Y_path, X_path, coords_path, device, normalize_X, normalize_Y)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for idx in dataloader:
        print(idx)
        break
