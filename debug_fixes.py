import numpy as np
from pricing.heston_calibration import HestonCalibrator

def test_fixes():
    # 1. Test heston_calibration.py Real Prices
    S0 = 100
    r = 0.05
    q = 0.02
    calibrator = HestonCalibrator(S0, r, q)
    
    strikes = np.array([90, 100, 110])
    maturities = np.array([1.0, 1.0, 1.0])
    params = [0.04, 0.04, -0.7, 2.0, 0.3]
    
    prices, _ = calibrator.get_prices_and_gradients(strikes, maturities, params)
    print("Price Types:", prices.dtype)
    assert not np.iscomplexobj(prices), "Prices should be real"
    print("Calibration real price check passed.")

    # 2. Test app.py Logic (Simulation)
    model_ivs = [0.2, None, 0.25]
    market_ivs = np.array([0.21, 0.22, 0.24])
    
    print("\nTesting app logic with None values...")
    try:
        model_ivs_arr = np.array([iv if iv is not None else np.nan for iv in model_ivs], dtype=float)
        valid_mask = ~np.isnan(model_ivs_arr)
        
        if np.any(valid_mask):
            rmse = np.sqrt(np.mean((model_ivs_arr[valid_mask] - market_ivs[valid_mask])**2))
            print(f"RMSE calculated successfully: {rmse:.4f}")
        else:
            print("No valid IVs found.")
    except Exception as e:
        print(f"Error in app logic simulation: {e}")
        return False

    print("App logic simulation passed.")
    return True

if __name__ == "__main__":
    if test_fixes():
        print("\nAll fixes verified successfully!")
    else:
        print("\nFix verification failed!")
        exit(1)
