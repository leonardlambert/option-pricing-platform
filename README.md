# Options Pricing & Volatility Analysis App

This application is a desk-oriented options analytics tool focused on **pricing, valuation, and middle-office use cases**.  
It emphasizes robustness, numerical stability, and realistic market constraints rather than academic sophistication.

The project is designed to be **usable immediately** (no setup friction) while still exposing deeper quantitative and engineering work for those who want to inspect it.

---

## Scope & Features

### Core Functionality (default)
- **Option pricing & Greeks**
  - Black–Scholes pricing
  - Robust implied volatility solver (Newton + bisection fallback)
  - Explicit handling of edge cases (low vega, bounds, expiry)

- **Strategy builder / spread visualization**
  - Payoff at maturity
  - Current value via model pricing
  - Greeks at strategy level

- **Monte Carlo PnL visualization**
  - Forward price simulation
  - PnL distributions
  - Scenario-based inspection

- **Volatility smile**
  - Sparse market strikes (API-constrained)
  - Interpolation in log-moneyness
  - ATM IV extraction at spot
  - Toggle between strike and moneyness representation

- **Volatility surface** *(in progress)*
  - Normalized representation (log-moneyness × maturity)
  - Designed for stability and interpretability rather than over-smoothing

- **Risk / stress testing** *(in progress)*

---

## Design Principles

- **Market realism over theory**
  - Handles missing strikes, low liquidity, solver failures
  - Makes no assumption that data is clean or complete

- **Separation of concerns**
  - Market data ingestion
  - Pricing and numerical routines
  - Visualization and UI

- **Safe defaults**
  - No exotic models enabled by default
  - No API key required to explore the app

---

## Demo Mode vs Live Data

The app runs in **Demo Mode by default**, using cached market data snapshots.

This avoids:
- API key requirements
- Setup friction
- External dependencies

Live market data can be enabled optionally by providing an API key in the UI.  
All core functionality works identically in demo mode.

---

## Advanced / Add-On Features

Some features are intentionally isolated as **optional add-ons**:
- Exotic pricing processes
- Alternative interpolation schemes
- Experimental numerical methods

These are provided as proof of:
- Quantitative depth
- Numerical robustness
- Software engineering practices

They are **off by default** and do not affect core outputs unless explicitly enabled.

---

## Conventions & Assumptions

- **Moneyness** is defined as:
  - `log(K / S)` (or `log(K / F)` when available)
- Smiles and surfaces are normalized to avoid spot-induced distortions
- OTM options are preferred when building smiles

---

## Intended Audience

- Pricing / valuation roles
- Middle-office
- Risk and product support
- Quant-adjacent engineering roles

This is **not** a research or calibration library.

---

## Documentation & References

Mathematical derivations, model assumptions, and numerical notes are available as linked PDFs for reference and auditability, without cluttering the UI.

---

## Disclaimer

This project is for **educational and demonstration purposes only**.  
It is not intended for live trading or production use.
