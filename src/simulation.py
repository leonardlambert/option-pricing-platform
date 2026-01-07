import numpy as np
from numba import njit

#Uses numba for faster simulations

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
def simulate_heston_paths(
    S0, r,
    v0, kappa, theta, xi, rho,
    T, steps, n_paths
):
    """
    Simulate Heston stochastic volatility paths using full truncation Euler.
    
    dS_t = r S_t dt + sqrt(v_t) S_t dW1_t
    dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW2_t
    corr(dW1, dW2) = rho
    """
    dt = T / steps
    paths = np.zeros((n_paths, steps + 1))
    vars_ = np.zeros((n_paths, steps + 1))

    paths[:, 0] = S0
    vars_[:, 0] = v0

    for i in range(n_paths):
        for t in range(1, steps + 1):
            z1 = np.random.standard_normal()
            z2 = np.random.standard_normal()

            # Correlated Brownian motions
            w1 = z1
            w2 = rho * z1 + np.sqrt(1 - rho**2) * z2

            v_prev = max(vars_[i, t-1], 0.0)

            # Variance process (full truncation)
            v_new = (
                v_prev
                + kappa * (theta - v_prev) * dt
                + xi * np.sqrt(v_prev * dt) * w2
            )
            v_new = max(v_new, 0.0)

            # Asset price
            paths[i, t] = paths[i, t-1] * np.exp(
                (r - 0.5 * v_prev) * dt
                + np.sqrt(v_prev * dt) * w1
            )

            vars_[i, t] = v_new

    return paths, vars_

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