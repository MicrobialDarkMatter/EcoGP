from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import torch
import tqdm

if __name__ == "__main__":
    # LOAD DATA
    from torch.utils.data import DataLoader, random_split
    from EcoGP.DataLoad import DataLoad
    from EcoGP.DataSampler import DataSampler

    from EcoGP.misc.save_results import save_results
    from EcoGP.misc.calculate_metrics_fast import calculate_metrics

    from sklearn import metrics

    # from configs.config_clean_unique import config  # TODO: Set config
    # from configs.config_butterfly import config  # Import the config module
    # from configs.config_central_park import config
    from configs.config_toy import config

    # ARGUMENTS
    # environment = config["additive"]["environment"]
    # spatial = config["additive"]["spatial"]
    # traits = config["additive"]["traits"]

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
    split_pct = config["general"]["split_pct"]
    n_inducing_points_env = config["environmental"]["n_inducing_points"]
    n_inducing_points_spatial = config["spatial"]["n_inducing_points"]

    verbose = config["general"]["verbose"]
    presence_absence = config["data"]["presence_absence"]
    normalize_X = config["data"]["normalize_X"]
    likelihood = config["general"]["likelihood"]
    seed = config["general"]["seed"]

    # prevalence_threshold = config["data"]["prevalence_threshold"]
    # STOP ARGUMENTS

    res = {
        "ROC AUC": [],
        "PR AUC": [],
        "NLL": [],
        "MAE": [],
    }
    for seed in range(0, 5):
        data = DataLoad(
            Y_path=y_path,
            X_path=x_path,
            coords_path=coords_path,
            traits_path=traits_path,
            device=device,
            normalize_X=normalize_X,
            total_counts_path="",
            presence_absence_Y=presence_absence,
            verbose=verbose
        )

        dataset = DataSampler(data)

        if coords_path:
            train_indices, validation_indices, test_indices = random_split(torch.arange(dataset.unique_coords.shape[0]),
                                                                           split_pct,
                                                                           generator=torch.Generator().manual_seed(
                                                                               seed))

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
                                                                           generator=torch.Generator().manual_seed(
                                                                               seed))

        # Make sure at least 10 species obserservations are present in each subset of the data
        keep_y = (dataset.Y[train_dataset.indices].sum(dim=0) >= split_pct[0] * 10) & (
                dataset.Y[validation_dataset.indices].sum(dim=0) >= split_pct[1] * 10) & (
                         dataset.Y[test_dataset.indices].sum(dim=0) >= split_pct[2] * 10)
        dataset.Y = dataset.Y[:, keep_y]
        dataset.taxon_names = dataset.taxon_names[keep_y]
        dataset.n_species = dataset.Y.shape[1]
        if traits_path:
            dataset.traits = dataset.traits[keep_y, :]
        print(f"Keeping {keep_y.sum().item()} taxons after at least one observation in each split")

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

        preds = []

        for j in tqdm.tqdm(range(dataset.n_species)):
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

        metrics = calculate_metrics(test_Y, y_prob)

        print(metrics)

        res["ROC AUC"].append(metrics["AUC"])
        res["NLL"].append(metrics["NLL"])
        res["MAE"].append(metrics["MAE"])
        res["PR AUC"].append(metrics["PR_AUC"])

    for key, value in res.items():
        print(key, torch.tensor(value).mean(), torch.tensor(value).std())

