# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python financial engineering project for calibrating stochastic models to market option data and pricing derivatives. It is part of a KU Leuven Master's in Actuarial Science coursework.

## Running the Code

```bash
# Install dependencies
pip install numpy pandas matplotlib scipy

# Run the script
python main.py
```

No build, lint, or test tooling is configured.

## Architecture

The entire project is a single class, **`DerivativePricer`**, in `main.py`. It takes market data (option prices, strikes, maturities) and market parameters (S0, r, q) on construction.

### Pricing Pipeline

1. **Calibration** — `calibrate_model(model, starting_values, hyper_params)` fits model parameters to market prices. It calls `_generate_loss_function()` to build an MSE loss over FFT-computed prices, then calls `_optimize()` which runs Nelder-Mead from multiple starting points to avoid local minima.

2. **FFT Pricing** — `_fft(char_func, T, N, alpha, eta)` implements the Carr-Madan algorithm: computes a damped call price via FFT over characteristic function values (Simpson's rule integration), then interpolates at market strikes.

3. **Monte Carlo** — `Monte_Carlo(maturity, steps, contract, contract_params, model, model_params)` simulates 10,000 price paths under a chosen model and prices a contract. The inner helper `calc_derivative_value(price_paths, contract, contract_params)` handles payoff computation.

### Supported Models

| Model | Params | Status |
|---|---|---|
| `'Black-Scholes'` | σ | Fully implemented |
| `'Heston'` | κ, η, θ, ρ, v₀ | Fully implemented |
| `'Bates'` | — | Stubbed (`pass`) |
| `'VG'` | — | Stubbed (`pass`) |

### Supported Contracts (Monte Carlo)

- `'European call'` / `'European put'` — implemented
- `'Variance swap'` — stubbed

## Known Issues / Incomplete Parts

- **Main block** (`__name__ == "__main__"`, lines 392–411): loads data and sets starting values but never calls any methods.
- **Bates and VG models**: loss functions and MC path generation are not implemented.
- **Variance swap payoff**: not implemented.
- **f-string bug** (line 81): `"{self.calibration_time:.3f}s"` is missing the `f` prefix.
- **Argument order mismatch** (line 386): `calc_derivative_value` is called with `(price_paths, contract_params, contract)` but defined as `(price_paths, contract, contract_params)`.

## Data

`Data.csv` contains market option prices with columns `Strikes`, `Maturities` (in years), and `Prices`. Used as the calibration target.
