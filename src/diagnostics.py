import numpy as np

def calculate_metrics(market_ivs, model_ivs):
    """
    Calculate common error metrics between market and model implied volatilities.
    """
    market_ivs = np.array(market_ivs)
    model_ivs = np.array(model_ivs)
    
    errors = market_ivs - model_ivs
    abs_errors = np.abs(errors)
    
    rmse = np.sqrt(np.mean(errors**2))
    mae = np.mean(abs_errors)
    mape = np.mean(abs_errors / market_ivs) * 100 # Percentage
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'MaxError': np.max(abs_errors)
    }

def check_parameter_stability(params):
    """
    Check for potential calibration instability or ill-posedness.
    """
    warnings = []
    
    # 1. Feller Condition
    kappa = params['kappa']
    theta = params['theta']
    xi = params['xi']
    feller = 2 * kappa * theta - xi**2
    if feller <= 0:
        warnings.append({
            'type': 'Feller Violation',
            'message': f"$2\\kappa\\theta$ ({2*kappa*theta:.4f}) < $\\xi^2$ ({xi**2:.4f}). Process can reach zero.",
            'severity': 'WARNING'
        })
        
    # 2. High Vol of Vol
    if xi > 2.0:
        warnings.append({
            'type': 'High Vol-of-Vol',
            'message': f"$\\xi$ ({xi:.2f}) is very high, which can lead to numerical instability and 'sharp' smiles.",
            'severity': 'CAUTION'
        })
        
    # 3. High Correlation
    if abs(params['rho']) > 0.95:
        warnings.append({
            'type': 'High Correlation',
            'message': f"$\\rho$ ({params['rho']:.2f}) is near the boundary, which may indicate over-fitting or ill-posedness.",
            'severity': 'NOTE'
        })
        
    return warnings
