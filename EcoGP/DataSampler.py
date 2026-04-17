import os.path
import warnings

import torch

from torch.utils.data import DataLoader, Dataset


class DataSampler(Dataset):
    def __init__(self, data):
        self.__dict__.update(data.__dict__)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return idx

    def get_batch_data(self, idx):
        """
        Returning data for batch indices
        :param idx: list of integers
        :return: Target data if available
        """
        X = self.X[idx].to(self.device) if self.X is not None else None
        Y = self.Y[idx].to(self.device) if self.Y is not None else None
        coords = self.coords[idx].to(self.device) if self.coords is not None else None
        traits = self.traits.to(self.device) if self.traits is not None else None

        return X, Y, coords, traits


if __name__ == "__main__":
    from DataLoad import DataLoad

    Y_path = "../data/clean/Y.csv"
    X_path = "../data/clean/X.csv"
    coords_path = "../data/clean/XY.csv"
    total_counts_path = "../data/clean/total_counts.csv"
    device = torch.device("cpu")
    normalize_X = True
    batch_size = 50

    data = DataLoad(Y_path, X_path, coords_path, device, normalize_X, verbose=True, total_counts_path=total_counts_path)
    datasampler = DataSampler(data)
