import numpy as np
from ctypes import c_float, c_int, Structure, POINTER, c_ulonglong
import ctypes
from scipy.io import savemat

def as_pointer(arr):
    if arr is None:
        return None
    assert arr.dtype == np.float32
    assert arr.flags['C_CONTIGUOUS'], "Array must be C_CONTIGUOUS"
    return arr.ctypes.data_as(POINTER(c_float))

def run_cuda_simulation(model, theta):

    n_trials    = c_int(model.cfg.n_trials)
    timesteps   = c_int(model.cfg.timesteps)
    maxTimeLag  = c_int(model.cfg.maxTimeLag)
    n_params    = c_int(model.cfg.n_params)
    n_ou        = c_int(model.cfg.n_ou)
    n_particles = c_int(theta.shape[1])
    mode        = c_int(model.cfg.mode)

    deltaT      = c_float(model.cfg.deltaT)

    real_acf    = np.ascontiguousarray(model.real_acf, dtype=np.float32)
    theta       = np.ascontiguousarray(theta, dtype=np.float32)

    OUTPUT_DISTANCE  = 1 << 0
    OUTPUT_AVG_ACF   = 1 << 1
    OUTPUT_TRIAL_ACF = 1 << 2
    OUTPUT_SIM_DATA  = 1 << 3

    # Prepare output arrays
    d               = (np.zeros(n_particles.value, dtype=np.float32).copy(order='C') if model.cfg.mode & OUTPUT_DISTANCE else None)
    syn_acf         = (np.zeros((n_particles.value, maxTimeLag.value), dtype=np.float32) if model.cfg.mode & OUTPUT_AVG_ACF else None)
    syn_acf_trials  = (np.zeros((n_particles.value, n_trials.value, maxTimeLag.value), dtype=np.float32).copy(order='C') if model.cfg.mode & OUTPUT_TRIAL_ACF else None)
    syn_data        = (np.zeros((n_particles.value, model.cfg.n_trials, model.cfg.timesteps), dtype=np.float32).copy(order='C') if model.cfg.mode & OUTPUT_SIM_DATA else None)

    d_ptr               = as_pointer(d)
    syn_acf_ptr         = as_pointer(syn_acf)
    syn_acf_trials_ptr  = as_pointer(syn_acf_trials)
    syn_data_ptr        = as_pointer(syn_data)
    real_acf_ptr        = as_pointer(real_acf)
    theta_ptr           = as_pointer(theta)

    params = model.OUParams(
        output_acf=syn_acf_ptr,
        output_distance=d_ptr,
        output_trial_acfs=syn_acf_trials_ptr,
        output_sim_data=syn_data_ptr,
        real_acf=real_acf_ptr,
        thetas=theta_ptr,
        n_particles=n_particles,
        n_params=n_params,
        n_trials=n_trials,
        timesteps=timesteps,
        max_lag=maxTimeLag,
        dt=deltaT,
        n_ou=n_ou,
        mode=mode,
        seed=0  # or generate here if needed
    )

    # Run CUDA kernel
    model.run_fn(ctypes.byref(params))

    # savemat('sim_outputs.mat',{'output_distance': output_distance, 
    #                            'output_acf':output_acf, 
    #                            'output_trial_acfs':output_trial_acfs, 
    #                            'output_sim_data':output_sim_data})


    return d, syn_acf, syn_acf_trials, syn_data
