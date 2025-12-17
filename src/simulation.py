import numpy as np
from numba import njit

@njit
def simulate_gbm_paths(S0, r, sigma, T, steps, n_paths):
    """
    Simulate Geometric Brownian Motion paths.
    """
    dt = T / steps
    paths = np.zeros((n_paths, steps + 1))
    paths[:, 0] = S0
    
    for i in range(n_paths):
        for t in range(1, steps + 1):
            z = np.random.standard_normal()
            paths[i, t] = paths[i, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
            
    return paths

@njit
def simulate_mjd_paths(S0, r, sigma, lamb, mu_j, sigma_j, T, steps, n_paths):
    """
    Simulate Merton Jump Diffusion paths.
    """
    dt = T / steps
    paths = np.zeros((n_paths, steps + 1))
    paths[:, 0] = S0
    
    for i in range(n_paths):
        for t in range(1, steps + 1):
            z = np.random.standard_normal()
            n_jumps = np.random.poisson(lamb * dt)
            jump_factor = 0.0
            if n_jumps > 0:
                for _ in range(n_jumps):
                    jump_factor += np.random.normal(mu_j, sigma_j)
            
            paths[i, t] = paths[i, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z + jump_factor)
            
    return paths

@njit
def simulate_vg_paths(S0, r, sigma, theta, nu, T, steps, n_paths):
    """
    Simulate Variance Gamma paths.
    """
    dt = T / steps
    paths = np.zeros((n_paths, steps + 1))
    paths[:, 0] = S0
    
    for i in range(n_paths):
        for t in range(1, steps + 1):
            # Gamma process for time change
            g = np.random.gamma(dt/nu, nu)
            z = np.random.standard_normal()
            
            # VG increment
            dX = theta * g + sigma * np.sqrt(g) * z
            
            # Risk neutral correction (martingale adjustment)
            # omega = (1/nu) * ln(1 - theta*nu - sigma^2*nu/2)
            # But commonly approximate/drift adjusted directly in price
            # Here we follow a standard implementation:
            # S_t = S_0 * exp(r*t + X_t + omega*t)
            omega = (1/nu) * np.log(1 - theta*nu - 0.5*sigma**2*nu)
            
            paths[i, t] = paths[i, t-1] * np.exp(r*dt + dX + omega*dt)
            
    return paths
