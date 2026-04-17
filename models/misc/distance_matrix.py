import torch
import numpy as np


def get_distance_matrix(coordinates):
    """
    Get distance matrix for all points in xy with longitude and latitude, respectively.

    :param coordinates: as tensor
    :return: distance matrix tensor
    """
    RADIUS = 6373  # Approximate radius of Earth in km

    # Convert degrees to radians
    lon, lat = map(torch.deg2rad, (coordinates[:, 0], coordinates[:, 1]))

    # Compute differences
    dlon = lon - lon.unsqueeze(-1)  # Shape: (N, M, K)
    dlat = lat - lat.unsqueeze(-1)  # Shape: (N, M, K)

    # Haversine formula
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat.unsqueeze(-1)) * torch.cos(lat) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    # Compute distance
    return RADIUS * c


if __name__ == "__main__":
    __coords = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # longitude, latitude
    __dist_matrix = get_distance_matrix(__coords)
