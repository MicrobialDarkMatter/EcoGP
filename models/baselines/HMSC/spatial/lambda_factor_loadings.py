import torch

import pyro
import pyro.distributions as dist
import torch.distributions.constraints as constraints


def lambdas_spatial_model(batch, n_latents: int):
    # Note: Lambda as h,j and not j,h!!! (Possible error in HMSC paper)

    device = batch.get("device")
    n_species = batch.get("n_species")

    latent_plate = pyro.plate("latent_plate", n_latents, dim=-1)
    species_plate = pyro.plate("_species_plate", n_species, dim=-2)

    # TODO: Possibly hyperparameters and not learnable
    v = torch.tensor(1.0, device=device)  # pyro.param("v", torch.tensor(1.0), constraint=constraints.positive).to(device)
    a1 = torch.tensor(1.0, device=device)  # pyro.param("a1", torch.tensor(1.0), constraint=constraints.positive).to(device)
    a2 = torch.tensor(1.0, device=device)  # pyro.param("a2", torch.tensor(1.0), constraint=constraints.positive).to(device)

    # Tau
    delta1 = pyro.sample(f"delta_1", dist.Gamma(a1, 1.)).reshape(-1)
    if n_latents > 1:
        with pyro.plate("delta2_plate", n_latents - 1, dim=-1):
            delta2_up = pyro.sample(f"delta_2_up", dist.Gamma(concentration=torch.full((n_latents - 1,), a2, device=device),
                                                              rate=torch.full((n_latents - 1,), 1., device=device)))

        tau = torch.cumprod(torch.cat((delta1, delta2_up), dim=0), dim=0)
    else:
        tau = delta1

    with species_plate, latent_plate:
        phi = pyro.sample("phi", dist.Gamma(v / 2, v / 2))

        lambdas = pyro.sample("lambdas", dist.Normal(torch.zeros(n_species, n_latents, device=device), phi ** -1 * tau ** -1))

    return lambdas
