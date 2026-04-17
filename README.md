# Biogeography
*Beyond Occurrences: Bayesian Gaussian Processes for Relative Prevalence Species Distribution Modeling*

## Overview

Biogeography is a Python package for modeling species distributions using Gaussian Processes (GPs). The core model, **EcoGP**, leverages environmental and spatial focused GPs to capture complex ecological dependencies in microbial species data. The package is designed for scalable variational inference.
For more details, refer to our [paper (NO LINK YET)]().
---

## Features

- **EcoGP**: Multitask Gaussian Process model for species distribution modeling.
- **Training**: Variational inference with batch learning.
- **Configs**: Easily configurable config files for quick implementation on own data.
- **Baselines**: Standard models for comparison.
- **Dataset Support**: Ready-to-use datasets for butterflies, Central Park, and toy examples.

---

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/MicrobialDarkMatter/biogeography.git
   cd biogeography
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

---

## Project Structure

- [`EcoGP/model.py`](EcoGP/model.py): Main GP model.
- [`EcoGP/train.py`](EcoGP/train.py): Training script.
- [`configs/`](configs/): Configuration files for experiments.
- [`EcoGP/baselines/`](EcoGP/baselines/): Baseline models.
- [`data/`](data/): Datasets.

---

## Usage

### Training

Run the training script with a configuration file:

```sh
python EcoGP/train.py --config configs/config.py
```

- Replace [`config.py`](configs/config.py) with your desired config file.

### Configuration

- You can modify hyperparameters, dataset paths, and model options.

---

## EcoGP Model Details

- Variational setup using Pyro.
- Implements a multitask variational GP using GPyTorch.
- Supports batch learning.
- Option to customize kernels.

---

## Datasets

- Place your datasets in the [`data/`](data/) directory.
- Supported datasets: butterflies, Central Park, toy examples.

---

## Baselines

- Baseline models are in [`EcoGP/baselines/`](EcoGP/baselines/).
- Used these for comparison with EcoGP.

---

## Requirements

- Python 3.11+
- See [`requirements.txt`](requirements.txt) for all dependencies.

---

## Citation

If you use this package in your research, please cite:

```
TO COME
```

---

## License

See [`LICENSE`](LICENSE) for details.
