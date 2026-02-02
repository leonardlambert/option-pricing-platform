import numpy as np
from pricing.heston_calibration import HestonCalibrator

def test_calibration_logic():
    # Parameters for testing
    S0 = 100
    r = 0.05
    q = 0.02
    
    calibrator = HestonCalibrator(S0, r, q)
    
    # Dummy market data (M >= N=5)
    strikes = [90, 95, 100, 105, 110, 115]
    maturities = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    market_prices = [15.0, 12.5, 10.0, 7.5, 5.0, 3.0]
    
    # Initial guess [v0, vbar, rho, kappa, sigma]
    initial_guess = [0.04, 0.04, -0.7, 2.0, 0.3]
    
    print("Testing get_prices_and_gradients...")
    try:
        prices, jacobian = calibrator.get_prices_and_gradients(
            np.array(strikes), np.array(maturities), initial_guess
        )
        print("Prices:", prices)
        print("Jacobian shape:", jacobian.shape)
        print("Unpacking successful!")
    except Exception as e:
        print(f"Error in get_prices_and_gradients: {e}")
        return False

    print("\nTesting calibration loop...")
    try:
        res = calibrator.calibration(market_prices, strikes, maturities, initial_guess)
        print("Calibration results:")
        print("x:", res.x)
        print("Success:", res.success)
        print("Calibration successful!")
    except Exception as e:
        print(f"Error in calibration: {e}")
        return False
    
    return True

if __name__ == "__main__":
    if test_calibration_logic():
        print("\nAll tests passed!")
    else:
        print("\nTests failed!")
        exit(1)
