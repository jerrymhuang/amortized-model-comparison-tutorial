import numpy as np
from numba import njit, prange


@njit
def simulate_ddm_trial(
    v: float,
    a: float,
    z: float,
    tau: float,
    sigma: float = 1.0,
    dt: float = 0.001,
    max_steps: int = 10000,
    log_transform: bool = True,
) -> np.ndarray:

    if log_transform:
        a = np.exp(a)
        tau = np.exp(tau)

    # initialize
    # Starting point is proportionally scaled by sampled boundary
    x = z * a
    t = 0
    
    for _ in range(max_steps):
        t += dt
        x += v * dt + sigma * np.sqrt(dt) * np.random.normal()
        if x >= a:
            return np.array([t, 1.0], dtype=np.float32)
        if x <= -a:
            return np.array([t, 0.0], dtype=np.float32)
    # No decision within max_steps
    return np.array([-1.0, -1.0], dtype=np.float32)

def simulate_ddm(
    v: float,
    a: float,
    tau: float,
    z: float,
    sigma: float = 1.0,
    dt: float = 0.001,
    max_steps: int = 10000,
    num_trials: int = 500
) -> np.ndarray:

    sim_data = np.zeros((num_trials, 2), dtype=np.float32)

    for i in prange(num_trials):
        sim_trial = simulate_ddm_trial(
            v=v,
            a=a,
            z=z,
            tau=tau,
            sigma=sigma,
            dt=dt,
            max_steps=max_steps,
        )
        sim_data[i] = sim_trial

    return dict(rts=sim_data[:,0][..., None], choices=sim_data[:,1][..., None])