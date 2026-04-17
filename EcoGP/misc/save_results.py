import datetime
import pandas as pd


def save_results(file, x_path=None, y_path=None, coords_path=None, traits_path=None, n_latents=None, n_iter=None,
                 n_particles=None, device=None, lr=None, batch_size=None, train_pct=None, n_inducing_points=None,
                 n_species=None, n_samples=None, n_env=None, n_traits=None, model_name=None, note=None, auc=None,
                 nll=None, mae=None, path_figs=None, path_model=None):
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    data = {
        "x_path": x_path,
        "y_path": y_path,
        "coords_path": coords_path,
        "traits_path": traits_path,
        "n_latents": n_latents,
        "n_iter": n_iter,
        "n_particles": n_particles,
        "device": device,
        "lr": lr,
        "batch_size": batch_size,
        "train_pct": train_pct,
        "n_inducing_points": n_inducing_points,
        "n_species": n_species,
        "n_samples": n_samples,
        "n_env": n_env,
        "n_traits": n_traits,
        "model_name": model_name,
        "note": note,
        "auc": auc,
        "nll": nll,
        "mae": mae,
        "path_figs": path_figs,
        "path_model": path_model,
        "timestamp": time_stamp,
    }

    df = pd.read_excel(file)
    if df.empty:
        df = pd.DataFrame([data])
    else:
        df.loc[len(df)] = data

    df.to_excel(file, index=False)

# save_results("/Users/cp68wp/Documents/GitHub/HMSC-python/results/run_results.xlsx", x_path, y_path, coords_path, traits_path, n_latents, n_iter, n_particles, device, lr, batch_size, train_pct, n_inducing_points, n_species=None, n_samples=None, n_env=None, n_traits=None, model_name="Test", note="Notes", auc="", nll="", mae="", path_figs="", path_model="")