from .samplers import sample_from_prior, sample_from_pmc
from .simulator import run_cuda_simulation
from .filter import filter_by_epsilon
import numpy as np
import time

def basic_abc(model, pmc_mode):
    accepted_count, trial_count = 0, 0
    accepted_theta_step, accepted_d_step = np.empty((model.cfg.n_params, 0)), np.empty((1, 0))

    while accepted_count < model.cfg.min_samples:
        trial_count += model.cfg.min_samples

        start_time = time.time()
        if pmc_mode:
            theta = sample_from_pmc(model)
        else:
            theta = model.draw_theta_batch(n_samples=model.cfg.min_samples)
        stop_time = time.time() - start_time

        d, syn_acf, syn_acf_trials, syn_data = run_cuda_simulation(model, theta)
        accepted_theta_tmp, accepted_d_tmp = filter_by_epsilon(theta, d, model.epsilon[-1])

        if accepted_theta_tmp.shape[1] > 0:
            accepted_theta_step = np.hstack((accepted_theta_step, accepted_theta_tmp))
            accepted_d_step = np.hstack((accepted_d_step, accepted_d_tmp.reshape(1, -1)))
            accepted_count += accepted_theta_tmp.shape[1]
    
    model.accepted_theta.append(accepted_theta_step)
    model.accepted_d.append(accepted_d_step)

    model.accepted_count.append(accepted_count)
    model.total_count.append(trial_count)

    return d, syn_acf, syn_acf_trials, syn_data
