import numpy as np
from scipy.fft import fft

def log_strike_partition(eta=0.25, N=4096):
    """
    Creates a partition of strike prices in the log-space.
    Matches logic from optionfft.py.
    """
    b = np.pi / eta
    lamb = 2 * np.pi / (eta * N)
    k = -b + lamb * np.arange(0, N)
    return b, lamb, k

def fft_pricer(K, S0, T, r, char_func, args=(), alpha=1.5, N=4096, eta=0.25, call=True):
    """
    Pricing using Fast Fourier Transform (Carr-Madan 1999).
    Adapted from resources/OptionFFT/optionfft.py.
    
    Args:
        K (float): Target Strike (used for interpolation)
        S0 (float): Spot Price
        T (float): Time to Maturity
        r (float): Risk-free rate
        char_func (callable): Function with signature (u, T, r, ...args, S0)
        args (tuple): Additional arguments for char_func (excluding S0)
        alpha (float): Damping factor
        N (int): Grid size (power of 2)
        eta (float): Grid spacing
    """
    # Create integration partition
    V = np.arange(0, N * eta, eta)
    
    # Create log-strike partition
    b, lamb, k = log_strike_partition(eta, N)
    
    # Simpson's Rule Weights
    pm_one = np.empty((N,))
    pm_one[::2] = -1
    pm_one[1::2] = 1
    weights = 3 + pm_one
    weights[0] -= 1
    weights = (eta / 3) * weights
    
    # Calculate Modified Call Fourier Transform
    # Psi_T(v) = exp(-rT) * phi(v - (alpha+1)i) / (alpha^2 + alpha - v^2 + i(2alpha+1)v)
    v = V
    complex_u = v - (alpha + 1) * 1j
    
    # Call char_func. NOTE: We pass S0 as last argument to match our new convention
    # char_func signature: (u, T, r, *args, S0)
    phi_val = char_func(complex_u, T, r, *args, S0)
    
    denom = (alpha**2 + alpha - v**2) + (2*alpha + 1) * v * 1j
    psi = np.exp(-r * T) * phi_val / denom
    
    # FFT
    x = np.exp(1j * b * V) * psi * weights
    fft_vals = fft(x)
    call_prices_grid = np.real((np.exp(-alpha * k) / np.pi) * fft_vals)
    
    # Interpolate for specific K
    log_K = np.log(K)
    strikes_grid = np.exp(k)
    
    price = np.interp(K, strikes_grid, call_prices_grid)
    
    if not call:
        # Put-Call Parity: P = C - S + K * exp(-rT)
        price = price - S0 + K * np.exp(-r * T)
        
    return price
