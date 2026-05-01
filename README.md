# Option Pricer

A Python toolkit for pricing and calibrating derivatives using analytical, Monte Carlo, and model-free methods. Supports Black-Scholes and Heston stochastic volatility models.

---

## Features

- **FFT-based pricing** via Carr-Madan method for fast European option valuation
- **Model calibration** using multi-start Nelder-Mead optimization
- **Monte Carlo simulation** for European options and variance swaps
- **Model-free variance estimation** (VIX replication / vixification)
- **Put-call parity** to derive missing option legs

### Supported Models

| Model | Parameters |
|---|---|
| Black-Scholes | σ (volatility) |
| Heston | κ, θ, η, ρ, V₀ |

---

## Getting Started

### Prerequisites

- Python 3.9+
- pip or conda

### Installation

```bash
git clone https://github.com/<your-username>/Option-Pricer.git
cd Option-Pricer
pip install -r requirements.txt
```

### Data Format

The pricer expects a CSV file with three columns:

```
Strikes, Maturities, Prices
90, 0.25, 12.5
100, 0.25, 5.2
...
```

The included `Data.csv` can be used as a reference or replaced with your own market data.

---

## Usage

```python
import pandas as pd
from pricer import DerivativePricer

# Load market data
data = pd.read_csv("Data.csv")

pricer = DerivativePricer(
    option_prices=data["Prices"],
    strikes=data["Strikes"],
    maturities=data["Maturities"],
    S0=100,    # spot price
    r=0.05,    # risk-free rate
    q=0.0,     # dividend yield
)

# Calibrate Heston model
starting_values = [0.5, 0.04, 0.3, -0.7, 0.04]
hyper_params = {"N": 4096, "alpha": 1.5, "eta": 0.25}
params, loss = pricer.calibrate_model("heston", starting_values, hyper_params)

# Price options via FFT
char_func = pricer._build_char_func("heston", params)
prices = pricer.fft_pricer(char_func, T=0.5, strikes=[95, 100, 105], N=4096, alpha=1.5, eta=0.25)

# Estimate fair variance swap strike
var_strike = pricer.vixification(T=0.5, strikes=..., call_prices=..., put_prices=...)
```

Run the full example:

```bash
python pricer.py
```

---

## Project Structure

```
Option-Pricer/
├── pricer.py          # Main library (DerivativePricer class)
├── Data.csv           # Sample market data (strikes, maturities, prices)
├── Paper.pdf          # Reference paper with theoretical background
├── requirements.txt   # Python dependencies
└── README.md
```

---

## Theoretical Background

The implementation is based on:

- **Carr & Madan (1999)** — FFT option pricing via characteristic functions
- **Heston (1993)** — Stochastic volatility model
- **CBOE VIX methodology** — Model-free variance swap replication

See `Paper.pdf` for a detailed derivation of the methods.


