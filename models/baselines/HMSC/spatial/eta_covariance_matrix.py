import torch


def get_eta_covariance_matrix(dist_matrix, lengthscale=torch.ones(1)):
    """
    Return shape (n_latents, n_locs, n_locs), with n_latents = len(lengthscale)
    :param coords:
    :param lengthscale:
    :return:
    """

    assert len(lengthscale) > 0, ValueError("Lengthscale must have at least one element")

    # Reshape lengthscale to match the new dimension to broadcast properly and create a new dimension for distance
    covariance = torch.exp(-dist_matrix.unsqueeze(0) / lengthscale.view(-1, 1, 1))

    return covariance + (torch.eye(dist_matrix.shape[0]) * 1e-5).unsqueeze(0)


if __name__ == "__main__":
    __dist_matrix = torch.tensor([[0., 1., 3.], [1., 0., 6.], [3., 6., 0.]])
    __lengthscale = torch.tensor([15., 5.])

    __eta_cov_matrix = get_eta_covariance_matrix(dist_matrix=__dist_matrix, lengthscale=__lengthscale)
    print(__eta_cov_matrix[0,:,:])
