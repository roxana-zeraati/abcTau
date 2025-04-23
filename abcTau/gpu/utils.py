import numpy as np
from scipy.stats import multivariate_normal
from scipy.io import savemat

def calc_weights(model):
    theta = model.accepted_theta[-1]   # (n_params, M)
    theta_prev = model.accepted_theta[-2]  # (n_params, N)
    weights = model.weights[-1]             # (N,)
    prior = model.cfg.prior
    tau_squared = model.tau_squared[-1]

    n_params, M = theta.shape
    _, N = theta_prev.shape

    # Evaluate prior probability for new samples
    prior_prob = np.array([
        np.prod([prior[j].pdf(theta[j, i]) for j in range(n_params)])
        for i in range(M)
    ])

    # Evaluate proposal probability (PMC mixture)
    if np.isscalar(tau_squared):
        cov = tau_squared * np.eye(n_params)
    else:
        cov = tau_squared

    # Compute mixture density: sum_j w_j * N(theta_i | theta_prev_j, tau²)
    proposal_prob = np.zeros(M)
    for j in range(N):
        mvn = multivariate_normal(mean=theta_prev[:, j], cov=cov)
        proposal_prob += weights[j] * mvn.pdf(theta.T)

    # Final importance weights
    weights_new = prior_prob / proposal_prob
    weights_new /= weights_new.sum()

    assert np.all(np.isfinite(weights_new)), "Invalid weights (NaN or Inf)"
    assert np.isclose(weights_new.sum(), 1.0, atol=1e-6), "weights should sum to 1"

    return weights_new


def effective_sample_size(w):
    return 1.0 / np.sum(w ** 2)

def weighted_covar(x, w):
    if x.shape[1] != len(w):
        raise ValueError("Length of weights not compatible with specified axis.")

    x_mean = np.average(x, axis=1, weights=w)
    xm = x - x_mean[:, np.newaxis]
    return (xm * w[np.newaxis, :]) @ xm.T / (1.0 - np.sum(w ** 2))


def ensure_theta_shape(theta, n_params):
    theta = np.atleast_2d(theta)
    if theta.shape[0] != n_params:
        theta = theta.T
    assert theta.shape[0] == n_params, f"Expected shape (n_params, N), got {theta.shape}"
    return theta

import numpy as np

def autocorr_fft(x, norm=True):
    x = np.asarray(x)
    x -= np.mean(x)
    n = len(x)
    fft = np.fft.fft(x, n=2*n)
    acf = np.fft.ifft(fft * np.conjugate(fft))[:n].real
    if norm:
        acf /= acf[0]
    return acf

def save_outputs(model, d, syn_acf, syn_acf_trials, syn_data):
    """
    Save all relevant ABC model and simulation outputs to .mat file.
    """
    def make_arrays_2d(x):
        if isinstance(x, list):
            for i, a in enumerate(x):
                x[i] = np.atleast_2d(a)
        return x
    
    def pad_array_list_with_nans(array_list):

        if not array_list:
            return np.array([], dtype=object)
        
        # Determine max shape across all arrays
        max_shape = np.array([a.shape for a in array_list]).max(axis=0)
        
        # Create padded versions
        padded_arrays = []
        for a in array_list:
            padded = np.full(max_shape, np.nan, dtype=np.float32)
            slices = tuple(slice(0, s) for s in a.shape)
            padded[slices] = a
            padded_arrays.append(padded)
        
        return np.array(padded_arrays)

    clean_tau_squared = np.array([
        float(x) if np.isscalar(x) else float(np.squeeze(x)) 
        for x in model.tau_squared
    ])


    outputs = {
        'accepted_theta': pad_array_list_with_nans(model.accepted_theta),
        'accepted_d': pad_array_list_with_nans(model.accepted_d),
        'accepted_count': np.array(model.accepted_count),
        'total_count': np.array(model.total_count),
        'epsilon': np.array(model.epsilon),
        'weights': pad_array_list_with_nans(model.weights),
        'tau_squared': clean_tau_squared,
        'eff_sample_size': np.array(model.eff_sample_size),
    }

    if d is not None:
        outputs['output_distance'] = d
    if syn_acf is not None:
        outputs['output_acf'] = syn_acf
    if syn_acf_trials is not None:
        outputs['output_trial_acfs'] = syn_acf_trials
    if syn_data is not None:
        outputs['output_sim_data'] = syn_data

    savemat(model.cfg.output_path, outputs, oned_as='column') # can be loaded in python with scipy.io.loadmat
    print(f"[INFO] Saved outputs to {model.cfg.output_path}")
