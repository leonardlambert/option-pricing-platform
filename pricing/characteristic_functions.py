import numpy as np

def phi_bsm(u, T, r, sigma, S0):
    """
    Characteristic function for Geometric Brownian Motion.
    Returns CF of log(S_T).
    """
    mu = np.log(S0) + (r - 0.5 * sigma**2) * T
    var = T * sigma**2
    return np.exp(1j * u * mu - 0.5 * u**2 * var)

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
    
    # Jump part: lambda * T * (phi_jump(u) - 1)
    # Jump size J ~ N(mu_j, sigma_j). ln(J) is Normal? No, J is jump in log price typically?
    # In Merton model: S_t = S_{t-} * Y. Y is Lognormal. X = ln(Y) ~ N(mu_j, sigma_j).
    # d(ln S) ... + ln(Y) dN.
    # CF of sum(ln Y) is exp(lambda T (E[e^{iu ln Y}] - 1))
    # E[e^{iu X}] = exp(iu mu_j - 0.5 sigma_j^2 u^2)
    
    jump_part = lamb * T * (np.exp(1j * u * mu_j - 0.5 * sigma_j**2 * u**2) - 1)
    
    # CAUTION: optionfft.py does slightly different formula check?
    # optionfft.py: jump_term = lamb * T * (np.exp(1j * u * mu_j - 0.5 * sigma_j**2 * u**2) - 1) 
    # That matches.
    # But optionfft.py definition of MJD? It doesn't have MJD class!
    # It has GeometricBrownianMotion and VarianceGamma. 
    # I should stick to my previous correct MJD formula but adapted for S0.
    
    return np.exp(term1 + term2 + jump_part)