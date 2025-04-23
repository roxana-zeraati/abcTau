import numpy as np
from scipy import stats
import cupy as cp

def sample_from_prior(model):
    return model.draw_theta()

def sample_from_pmc(model):
    weights = model.weights[-1]
    min_samples = model.cfg.min_samples
    theta_prev = model.accepted_theta[-1]
    tau_squared = model.tau_squared[-1]
    n_params = model.cfg.n_params
    n_particles = theta_prev.shape[1]

    # --- Draw indices from previous generation ---
    indices = np.random.choice(
        n_particles,
        size=min_samples * 10,  # oversample to ensure enough valid ones
        replace=True,
        p=weights / weights.sum()
    )
    theta_star = theta_prev[:, indices]  # shape: (n_params, oversampled)

    # --- Covariance ---
    if np.isscalar(tau_squared):
        cov = tau_squared * np.eye(n_params)
    else:
        cov = tau_squared

    # --- Draw perturbations in batch ---
    perturb = np.random.multivariate_normal(
        mean=np.zeros(n_params),
        cov=cov,
        size=theta_star.shape[1]
    ).T  # shape: (n_params, oversampled)

    proposals = theta_star + perturb  # shape: (n_params, oversampled)

    # --- Filter valid ones ---
    valid = proposals[:, np.all(proposals > 0, axis=0)]

    if valid.shape[1] < min_samples:
        print(f"[WARN] Only {valid.shape[1]} valid samples found, repeating...")
        return sample_from_pmc(model)  # try again recursively

    return valid[:, :min_samples]
