from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import torch

if __name__ == "__main__":
    # LOAD DATA
    from torch.utils.data import DataLoader, random_split
    from models.DataSampler import DataSampler

    from models.misc.save_results import save_results
    from models.misc.calculate_metrics import calculate_metrics

    from sklearn import metrics

    from configs.config import config  # TODO: Set config

    # ARGUMENTS
    environment = config["additive"]["environment"]
    spatial = config["additive"]["spatial"]
    traits = config["additive"]["traits"]

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
    train_pct = config["general"]["train_pct"]
    n_inducing_points_env = config["environmental"]["n_inducing_points"]
    n_inducing_points_spatial = config["spatial"]["n_inducing_points"]

    prevalence_threshold = config["data"]["prevalence_threshold"]
    # STOP ARGUMENTS

    dataset = DataSampler(
        Y_path=y_path,
        X_path=x_path,
        coords_path=coords_path,
        traits_path=traits_path,
        device=device,
        normalize_X=True,
        prevalence_threshold=prevalence_threshold)

    if spatial:
        train_indices, test_indices = random_split(torch.arange(dataset.unique_coords.shape[0]), [train_pct, 1 - train_pct], generator=torch.Generator().manual_seed(42))

        # Getting the spatial locations split into separate sets
        train_indices = dataset.coords_inverse_indicies[torch.isin(dataset.coords_inverse_indicies, torch.tensor(train_indices.indices))]
        test_indices = dataset.coords_inverse_indicies[torch.isin(dataset.coords_inverse_indicies, torch.tensor(test_indices.indices))]

        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)
    else:
        # dataloader = DataLoader(dataset=dataset, batch_size=_batch_size, shuffle=True)
        train_size = int(train_pct * len(dataset))
        test_size = len(dataset) - train_size

        train_dataset, test_dataset = random_split(dataset, [train_size, test_size],
                                                   generator=torch.Generator().manual_seed(42))

    # Make sure at least 10 species obserservations are present in each subset of the data
    keep_y = (dataset.Y[train_dataset.indices].sum(dim=0) >= 10) & (
            dataset.Y[test_dataset.indices].sum(dim=0) >= 10)
    dataset.Y = dataset.Y[:, keep_y]
    dataset.traits = dataset.traits[keep_y, :]
    dataset.n_species = dataset.Y.shape[1]

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    preds = []

    for j in range(dataset.n_species):
        if dataset.Y[train_dataset.indices, j].std() == 0:
            preds.append(np.full((len(test_dataset.indices),), dataset.Y[0, j]))
            continue
        # Create MaxEnt model (Logistic Regression with no regularization by default)
        model = LogisticRegression(solver='lbfgs', max_iter=n_iter)

        # Train model
        model.fit(dataset.X[train_dataset.indices], dataset.Y[train_dataset.indices, j])

        # Get predicted probabilities
        probs = model.predict_proba(dataset.X[test_dataset.indices])

        # Appending probability for predicting 1
        preds.append(probs[:, model.classes_.astype(bool)].squeeze())

    y_prob = torch.tensor(preds).T
    test_Y = dataset.Y[test_dataset.indices]

    metric_results = calculate_metrics(test_Y, y_prob)

    print(metric_results)

    if True:
        prevalence = test_Y.sum(dim=0) / test_Y.shape[0]
        bin0 = (prevalence <= 0.01)
        # bin1 = ((prevalence > 0.001) & (prevalence <= 0.01))
        bin2 = ((prevalence > 0.01) & (prevalence <= 0.1))
        bin3 = (prevalence > 0.1)

        metric_results = calculate_metrics(test_Y[:, bin0], y_prob[:, bin0])
        print("bin0", metric_results)
        # metric_results = calculate_metrics(test_Y[:, bin1], y_prob[:, bin1])
        # print("bin1", metric_results)
        metric_results = calculate_metrics(test_Y[:, bin2], y_prob[:, bin2])
        print("bin2", metric_results)
        metric_results = calculate_metrics(test_Y[:, bin3], y_prob[:, bin3])
        print("bin3", metric_results)

