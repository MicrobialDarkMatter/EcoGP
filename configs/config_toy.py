import os
import torch

# from configs.data_folder_path import data_folder_path
from configs.base_path import base_path
from EcoGP.likelihoods import DirichletMultinomialLikelihood, BernoulliLikelihood

config = {
    "data": {
        "X_path": os.path.join(base_path, "data/toydata/X.csv"),
        "Y_path": os.path.join(base_path, "data/toydata/Y.csv"),
        "coords_path": "",# os.path.join(base_path, "data/clean/XY.csv"),
        "traits_path": "",#os.path.join(base_path, "data/clean/traits.csv"),
        "normalize_X": True,
        # "prevalence_threshold": 0.0,
        "total_counts_path": os.path.join(base_path, "data/toydata/total_counts.csv"),
        # "hierarchy_path": os.path.join(base_path, "data/clean/genome_taxonomy.csv"),
        "presence_absence": False,
    },
    "general": {
        "likelihood": DirichletMultinomialLikelihood,
        "n_iter": 200,
        "n_particles": 1,
        "lr": 0.0025,
        "batch_size": 256,
        "split_pct": [0.7, 0.2, 0.1],  # Train/Test/Val
        "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        "verbose": True,
        "save_model_path": os.path.join(base_path, "results/saved_models/"),
        "seed": 0,
    },
    "environmental": {
        "n_latents": 10,
        "n_inducing_points": 200,
    },
    "spatial": {
        "n_latents": 5,
        "n_inducing_points": 500,
    },
    "hmsc": {
        "k_folds": 5,
        "cross_validation": False,
        "likelihood": "bernoulli",
    },
    "additive": {  # To specify if certain components should be included or omitted.
        "environment": True,
        "spatial": False,
        "traits": False,
    }
}
