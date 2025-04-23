#%%
# === GPU/CUDA acceleration description ===
# The CUDA kernel accelerates simulation of multiple parameter sets ("particles") for 1- or 2-timescale 
# Ornstein-Uhlenbeck (OU) processes in parallel. For each particle, it:
#   - Simulates multiple trials of the OU process.
#   - Computes the autocorrelation function (ACF) for each trial.
#   - Accumulates statistics such as trial-wise ACFs, average ACF, and distance to the observed ACF.
# Output behavior is controlled by a bitmask `mode`, allowing selective retrieval of desired results 
# (e.g., distances only, full trial data, or ACFs).

#%%
# === Imports ===
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import sys
from scipy.io import loadmat

# Add local modules to system path for import resolution
sys.path.extend(['.','./abcTau'])

# === Local project imports ===
from abcTau.gpu.pmc import pmc_abc                    # Population Monte Carlo ABC driver
from abcTau.gpu.model import CUDAModel                # Custom model wrapper using CUDA acceleration
from abcTau.gpu.config import ABCConfig               # Configuration dataclass for all model/ABC settings

#%%
# === Load synthetic observed data from a ground-truth 1-OU process ===
# This is the "real" data we will try to match via simulation and ABC inference
real_data = np.load('./example_data/OU_tau20_mean0_var1_rawData.npy')
n_trials, timesteps = real_data.shape  # Get data dimensions (n_trials x timesteps)

#%%
# === Define simulation output mode using bitmask flags ===
# Use bitwise OR ("|") to combine multiple outputs.
# Available flags:
#   OUTPUT_DISTANCE   – compute distance to real ACF
#   OUTPUT_AVG_ACF    – output trial-averaged ACF
#   OUTPUT_TRIAL_ACF  – output trial-by-trial ACFs
#   OUTPUT_SIM_DATA   – output raw simulated data
# Example: mode = OUTPUT_DISTANCE | OUTPUT_AVG_ACF | OUTPUT_TRIAL_ACF
OUTPUT_DISTANCE  = 1 << 0  # Compute distance between simulated and real ACF
OUTPUT_AVG_ACF   = 1 << 1  # Return trial-averaged ACF per particle
OUTPUT_TRIAL_ACF = 1 << 2  # Return ACFs for each individual trial
OUTPUT_SIM_DATA  = 1 << 3  # Return raw simulated data per trial
mode = OUTPUT_DISTANCE  # only output distances in this example

# === Define the generative model ===
# We are modeling the observed data using Ornstein-Uhlenbeck (OU) processes.
# Here we choose how many OU processes to include in the simulation:
# - n_ou = 1 uses a single-timescale OU process (simpler model)
# - n_ou = 2 would model data as a mixture of two OU processes (for multi-timescale dynamics)
n_ou = 1  # number of OU processes in the generative model

# === Prior over model parameters ===
# For the 1-OU model, there is a single time constant (tau).
# This prior assumes tau ~ Uniform(0, 100).
tau_min = 0.0
tau_max = 100.0
prior = [stats.uniform(loc=tau_min, scale=tau_max-tau_min)]

# === ABC Configuration ===
# This dataclass holds all relevant simulation, prior, and ABC settings
cfg = ABCConfig(
    real_data       = real_data,    # Raw observed data
    n_trials        = n_trials,     # Number of trials to simulate
    timesteps       = timesteps,    # Number of time steps per trial
    deltaT          = 1,            # Time resolution (ms)
    binSize         = 1,            # Binning size for analysis (ms)
    maxTimeLag      = 50,           # Max lag (in time bins) for ACF
    prior           = prior,        # Prior over parameters
    mode            = mode,         # Output modes from the CUDA kernel (bitmask)
    epsilon_0       = 1.0,          # Initial ABC acceptance threshold
    min_samples     = 100,          # Minimum accepted samples per iteration
    steps           = 60,           # Max number of PMC iterations
    n_ou            = n_ou,         # OU model type (1 or 2 processes)
    epsilon_floor   = 0.01,         # Terminate if epsilon drops below this
    output_path     = './outputs.mat'
)

# === Model instantiation and CUDA setup ===
# This object wraps the simulation kernel and prepares input/output buffers
model = CUDAModel(cfg)
model.load_cuda()  # Load the shared CUDA library and bind the simulation function

# === Run the PMC-ABC inference loop ===
# This will iteratively refine the posterior using simulation + distance rejection
pmc_abc(model)

# %% load outputs
# outputs = loadmat('./outputs.mat')
# %%
