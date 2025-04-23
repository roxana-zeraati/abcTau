from preprocessing import extract_stats
from summary_stats import comp_ac_fft
from .config import ABCConfig
import numpy as np
from simple_abc_general import Model
from ctypes import c_int, c_float, POINTER, Structure, CDLL

class OUParams(Structure):
    _fields_ = [
        ("output_acf", POINTER(c_float)),
        ("output_distance", POINTER(c_float)),
        ("output_trial_acfs", POINTER(c_float)),
        ("output_sim_data", POINTER(c_float)),
        ("real_acf", POINTER(c_float)),
        ("thetas", POINTER(c_float)),
        ("n_particles", c_int),
        ("n_params", c_int),
        ("n_trials", c_int),
        ("timesteps", c_int),
        ("max_lag", c_int),
        ("dt", c_float),
        ("n_ou", c_int),
        ("mode", c_int)
    ]


class CUDAModel(Model):
    def __init__(self, cfg: ABCConfig):
        super().__init__()
        self.cfg = cfg
        self.real_acf = np.ascontiguousarray(
            comp_ac_fft(cfg.real_data), dtype=np.float32
        )
        self.accepted_theta = []
        self.accepted_d = []
        self.accepted_count = []
        self.total_count = []
        self.epsilon = [self.cfg.epsilon_0]
        self.weights = []
        self.tau_squared = []
        self.eff_sample_size = []
        if self.cfg.n_ou == 1:
            self.cfg.n_params = 1
        elif self.cfg.n_ou == 2:
            self.cfg.n_params = 3

    def load_cuda(self):
        lib = CDLL("./abcTau/gpu/lib_ou_acf.so")
        fn = getattr(lib, "run_ou_n_acf")

        fn.argtypes = [POINTER(OUParams)]

        fn.restype = None

        self.OUParams = OUParams
        self.run_fn = fn
    
    def draw_theta_batch(self, n_samples):
        theta_samples = [p.rvs(size=n_samples) for p in self.cfg.prior]
        return np.stack(theta_samples, axis=0)
