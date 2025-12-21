# Options Pricing & Scenario Analysis Platform

This browser application is a desk-oriented options analytics tool focused on **pricing, valuation, and middle-office use cases**.  
It emphasizes robustness, numerical stability, and realistic market constraints rather than academic sophistication.

---

## Scope & Features

### Core Functionality
- **Option pricing & Greeks**
  - Pricing using Black–Scholes-Merton, Variance Gamma, and Merton models
  - Greeks calculation using closed-form expressions and finite differences

- **Strategy builder and Spread visualization**
  - Payoff at maturity & current value graph  
  - Multi-option strategy building and saving in an option book JSON file

- **PnL Distribution**
  - Monte Carlo Simulation using saved strategies, with adaptables path number / steps / horizon
  - Risk Metrics : Value at Risk (VaR), Conditional Value at Risk (CVaR)

- **Strategy Stress Testing**
  - Assessment of Spot / Volatility / Time Decay impact on PnL using Taylor approximation
  - Quick Scenario Analysis, PNL residual check

- **Volatility smile**
  - Construction using a preloaded dataset or live API data
  - Use of 5 strikes : one central, two OTM, two ITM

- **Volatility surface**
  - Plotting using a preloaded dataset or live API data
  - Normalized representation (log-moneyness × time to maturity)

---

## Demo Mode vs Live Data

The app runs in **Demo Mode by default**, using cached market data snapshots of AAPL, NVDA and SPY.

Live market data can be enabled optionally by providing an API key in the UI.  
All core functionality works identically in demo mode.

---

## Conventions & Assumptions

- **Moneyness** is defined as:
  - `log(K / S)` (or `log(K / F)` when available)
- Smiles are intepolated using a **cubic spline** / Surfaces are intepolated using a **griddata**

---

## Documentation & References

Mathematical derivations, model assumptions, and numerical notes are available as linked PDFs for reference and auditability, without cluttering the UI.

---

## Disclaimer

This project is for **educational and demonstration purposes only**.  
It is not intended for live trading or production use.
