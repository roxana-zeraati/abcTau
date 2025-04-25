from scipy import stats
import numpy as np
import time
from .abc_loop import basic_abc
from .utils import effective_sample_size, weighted_covar, calc_weights, save_outputs
from scipy.io import savemat

def pmc_abc(model):

    start_time = time.time()

    epsilon = model.epsilon[-1]

    for step in range(model.cfg.steps):

        print("\n" + "="*40)
        print(f"📦 PMC Step {step + 1}/{model.cfg.steps}")
        print("-" * 40)
        print(f"  ε (threshold)     : {epsilon:.5f}")

        step_start = time.time()

        d, syn_acf, syn_acf_trials, syn_data = basic_abc(model, pmc_mode=(step > 0))

        elapsed = time.time() - start_time
        
        epsilon = np.percentile(model.accepted_d[-1], 75)
        model.epsilon.append(epsilon)

        th = model.accepted_theta[-1]
        if step > 0:
            model.weights.append(calc_weights(model))
            model.tau_squared.append(2 * weighted_covar(th, model.weights[-1]))
        else:
            n_th = th.shape[1]
            w = np.ones(n_th) / n_th
            model.weights.append(w)
            model.tau_squared.append(2 * np.cov(th))

        model.eff_sample_size.append(effective_sample_size(model.weights[-1]))

        print(f"  Accepted samples  : {model.accepted_count[-1]} / {model.total_count[-1]}")
        print(f"  Acceptance rate   : {model.accepted_count[-1] / model.total_count[-1]:.2%}")
        print(f"  Effective Sample Size (ESS): {model.eff_sample_size[-1]:.2f}")
        print(f"  τ² (covariance scale)      : {model.tau_squared[-1] if np.isscalar(model.tau_squared[-1]) else model.tau_squared[-1].shape}")
        print(f"[PMC] Cumulative elapsed time: {elapsed:.2f} sec")
        print("="*40)

        if epsilon <= model.cfg.epsilon_floor:
            print(f"[PMC] Stopping early at step {step+1} — epsilon ({epsilon:.4f}) ≤ epsilon_floor ({model.cfg.epsilon_floor})")
            break

    total_time = time.time() - start_time
    print(f"\n[PMC] Finished all steps in {total_time:.2f} seconds")
    del model.epsilon[0]
    save_outputs(model, d, syn_acf, syn_acf_trials, syn_data)