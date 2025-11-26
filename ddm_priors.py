import numpy as np
from numba import njit 
from scipy.stats import halfnorm 

@njit
def ddm_baseline_priors():
    return {
        "v":    np.random.gamma(2., 1.),
        "a":    np.random.normal(-.1, 0.3),
        "z":    np.random.beta(2, 2),
        "tau":  np.random.normal(-1.5, 0.3),
    }

@njit
def ddm_standard_priors():
    return {
        "v":    np.random.gamma(2., 1.),
        "a":    np.random.normal(-.1, 0.3),
        "z":    np.random.beta(2, 2),
        "tau":  np.random.normal(-1.5, 0.3),
    }

@njit
def ddm_advanced_priors():
    return {
        "v":    np.random.gamma(2., 1.),
        "a":    np.random.normal(-.1, 0.3),
        "z":    np.random.beta(2, 2),
        "tau":  np.random.normal(-1.5, 0.3),
    }
