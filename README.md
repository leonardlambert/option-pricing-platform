# Options Pricing & Scenario Analysis Platform

This browser application is an options analytics tool focused on **pricing, valuation, and middle-office use cases**.  

It emphasizes robustness, numerical stability, and realistic market constraints rather than academic sophistication.

---

## Scope & Features

### Core Functionality
- **Option pricing & Greeks**
  - Pricing using standard Black–Scholes-Merton, and exotic Variance Gamma / Merton Jump Diffusion models
  - Greeks calculation using closed-form expressions and finite differences

  ![alt text](https://github.com/leonardlambert/option-pricing-platform/raw/master/src/images/tab1.png 'Single Option Pricing & Greeks by finite differences')

- **Strategy builder and Spread visualization**
  - Payoff at maturity & current value graph  
  - Multi-option strategy building and saving in an option book JSON file

  ![alt text](https://github.com/leonardlambert/option-pricing-platform/raw/master/src/images/tab2.png 'Spread building and visualization')

- **PnL Distribution**
  - Monte Carlo Simulation using saved strategies, with adaptables path number / steps / horizon
  - Risk Metrics : Value at Risk (VaR), Conditional Value at Risk (CVaR)

- **Strategy Stress Testing**
  - Assessment of Spot / Volatility / Time Decay impact on PnL using Taylor approximation
  - Quick Scenario Analysis, PNL residual check

- **Volatility smile**
  - Construction using a preloaded dataset or live API data
  - Use of 5 strikes : one central, two OTM, two ITM

  ![alt text](https://github.com/leonardlambert/option-pricing-platform/raw/master/src/images/tab4.png 'Volatility smile construction using dataset or live API data')

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

## References

- Black, F. & Scholes, M. (1973). **The Pricing of Options and Corporate Liabilities**. *Journal of Political Economy*, 81(3), 637–654.  
  https://doi.org/10.1086/260062

- Carr, P. & Madan, D. (1999). **Option Valuation Using the Fast Fourier Transform**. *The Journal of Computational Finance*, 2(4), 61–73.  
  https://doi.org/10.21314/jcf.1999.043

- Xiao, S., Ma, S., Li, G., & Mukhopadhyay, S. K. (2016). **European Option Pricing With a Fast Fourier Transform Algorithm for Big Data Analysis**. *IEEE Transactions on Industrial Informatics*, 12(3), 1219–1231.  
  https://doi.org/10.1109/TII.2015.2500885

---

## Disclaimer

This project is for **educational and demonstration purposes only**.  
It is not intended for live trading or production use, and should not be used as a basis for any trading decisions. I am not responsible for any losses or damages resulting from the use of this application.
