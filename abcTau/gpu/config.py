from dataclasses import dataclass
from typing import List
from scipy.stats import rv_continuous
import numpy as np
from numpy.typing import NDArray
from scipy.stats._distn_infrastructure import rv_frozen

@dataclass
class ABCConfig:

    n_trials:       int
    timesteps:      int
    binSize:        int
    maxTimeLag:     int
    n_ou:           int
    min_samples:    int
    epsilon_floor:  float

    real_data:      float
    deltaT:         float

    prior:          List[rv_continuous]

    # Model/CUDA
    mode: int

    # ABC algorithm
    epsilon_0:      float
    steps:          int

    output_path:    str

    def __post_init__(self):

        # ---- real_data shape check ----
        expected_shape = (self.n_trials, self.timesteps)
        if self.real_data.shape != expected_shape:
            raise ValueError(
                f"real_data must have shape {expected_shape}, "
                f"but got {self.real_data.shape}"
            )
        
        if not isinstance(self.prior, list):
            raise TypeError("prior must be a list")

        for i, dist in enumerate(self.prior):
            if not isinstance(dist, rv_frozen):
                raise TypeError(f"prior[{i}] must be a frozen distribution (scipy.stats.rv_frozen)")
        

