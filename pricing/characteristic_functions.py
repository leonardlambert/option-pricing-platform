import numpy as np
from pricing.heston_calibration import HestonCalibrator

def phi_bsm(u, T, r, sigma, S0):
    """
    Characteristic function for Geometric Brownian Motion.
    Returns CF of log(S_T).
    """
    mu = np.log(S0) + (r - 0.5 * sigma**2) * T
    var = T * sigma**2
    return np.exp(1j * u * mu - 0.5 * u**2 * var)

def phi_heston(
    u, T, r,
    v0, kappa, theta, xi, rho,
    S0
):
    """
    Characteristic function of log(S_T) under the Heston model.
    """
    iu = 1j * u

    a = kappa * theta
    b = kappa
    sigma = xi

    d = np.sqrt((rho * sigma * iu - b)**2 + sigma**2 * (iu + u**2))
    g = (b - rho * sigma * iu - d) / (b - rho * sigma * iu + d)

    exp_dt = np.exp(-d * T)

    C = (
        r * iu * T
        + (a / sigma**2) * (
            (b - rho * sigma * iu - d) * T
            - 2.0 * np.log((1 - g * exp_dt) / (1 - g))
        )
    )

    D = (
        (b - rho * sigma * iu - d) / sigma**2
        * ((1 - exp_dt) / (1 - g * exp_dt))
    )

    return np.exp(
        C + D * v0 + iu * np.log(S0)
    )


def phi_heston_2017(v0, theta, rho, kappa, sigma, S0, T, r, u):
    calibrator = HestonCalibrator(S0, r)
    return calibrator.characteristic_function(u, T, v0, theta, rho, kappa, sigma)

def phi_vg(u, T, r, sigma, theta, nu, S0):
    """
    Characteristic function for Variance Gamma.
    Returns CF of log(S_T).
    """
    omega = (1 / nu) * np.log(1 - theta * nu - 0.5 * sigma**2 * nu)
    # CF of X_T
    cf_X = np.power((1 - 1j * theta * nu * u + 0.5 * sigma**2 * nu * u**2), -T/nu)
    # S_T = S0 * exp((r+omega)T + X_T)
    # log(S_T) = log(S0) + (r+omega)T + X_T
    return np.exp(1j * u * (np.log(S0) + (r + omega) * T)) * cf_X

def phi_merton(u, T, r, sigma, lamb, mu_j, sigma_j, S0):
    """
    Characteristic function for Merton Jump Diffusion.
    Returns CF of log(S_T).
    """
    kappa = np.exp(mu_j + 0.5 * sigma_j**2) - 1
    drift = r - lamb * kappa
    
    # log(S0) + drift*T part
    term1 = 1j * u * (np.log(S0) + drift * T)
    
    # Diffusion part: -0.5 * sigma^2 * u^2 * T
    term2 = -0.5 * sigma**2 * u**2 * T
    
    jump_part = lamb * T * (np.exp(1j * u * mu_j - 0.5 * sigma_j**2 * u**2) - 1)
    
    return np.exp(term1 + term2 + jump_part)


