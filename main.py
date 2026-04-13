# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 22:14:54 2026

@author: michi
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Callable
import time
import seaborn as sns

class DerivativePricer():
    def __init__(
            self,
            option_prices: np.ndarray,
            strikes: np.ndarray,
            maturities: np.ndarray,
            S0: float,
            r: float,
            q: float = 0):

        """
        Initialise the pricer with market data and fixed parameters.

        Parameters
        ----------
        option_prices : np.ndarray
            Flat array of observed market option prices.
        strikes : np.ndarray
            Flat array of strike prices, parallel to option_prices.
        maturities : np.ndarray
            Flat array of maturities (in years), parallel to option_prices.
        S0 : float
            Current underlying price.
        r : float
            Continuously compounded risk-free rate.
        q : float, optional
            Continuous dividend yield. Default is 0.

        Notes
        -----
        The flat arrays are grouped internally by unique maturity, so each
        maturity maps to its own aligned strike and price vectors.
        """
        self.S0 = S0
        self.r = r
        self.q = q

        # Group flat (price, strike) observations by unique maturity
        unique_T, idx = np.unique(maturities, return_inverse=True)
        self.maturities    = unique_T
        self.option_prices = [option_prices[idx == i] for i in range(len(unique_T))]
        self.strikes       = [strikes[idx == i]        for i in range(len(unique_T))]
         
        
    def calibrate_model(self,
                        model: str,
                        starting_values: np.ndarray,
                        hyper_params: list = [4096, 1.5, 0.25]):
        """
        Fit model parameters to observed market option prices.

        Parameters
        ----------
        model : str
            Pricing model to calibrate. One of 'black-scholes', 'heston',
            'bates', 'vg'.
        starting_values : np.ndarray
            n × p array of initial parameter guesses, where n is the number
            of starting points and p is the number of model parameters.
        hyper_params : list, optional
            FFT hyperparameters [N, alpha, eta]. Default is [4096, 1.5, 0.25].

        Returns
        -------
        dict
            Summary containing optimal parameters, loss score, calibration
            time, and optimizer metadata (iterations, evaluations, message).

        Notes
        -----
        Runs Nelder-Mead from each row of starting_values independently and
        keeps the result with the lowest final loss. Stores optimal parameters
        in self.optimal_params.
        """
        
        
        t_start = time.time()
        
        self.model = model
        
        self.loss_function = self._generate_loss_function(self.model, hyper_params)
        
        self.optimal_params, self.optimize_result = self._optimize(starting_values)
        
        t_end = time.time()
        
        self.calibration_time = t_end - t_start
        
        summary = {
        "Optimal Parameters": ", ".join(f"{x:.2f}" for x in self.optimal_params),
        "Score": f"{self.optimize_result['score']:.2f}",
        "Calibration Time": f"{self.calibration_time:.3f}s",
        "Optimizer Info": {
            "Iterations": self.optimize_result["nit"],
            "Function Evaluations": self.optimize_result["nfev"],
            "Convergence Message": self.optimize_result["message"]
            }
        }
    
        print("===== Calibration Summary =====")
        for key, value in summary.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for subkey, subval in value.items():
                    print(f"  {subkey}: {subval}")
            else:
                print(f"{key}: {value}")
        print("===============================")
        
        return summary
 
    
    def _build_char_func(self, model: str, T: float, params: np.ndarray) -> Callable:
        """
        Return the characteristic function φ(u) for a given model, maturity, and parameters.

        Parameters
        ----------
        model : str
            Pricing model name (case-insensitive). One of 'black-scholes', 'heston'.
        T : float
            Time to maturity in years.
        params : np.ndarray
            Model parameter vector.

        Returns
        -------
        Callable
            φ(u) -> complex np.ndarray, the characteristic function of the log-price.
        """
        if model == 'black-scholes':
            sigma, = params
            return lambda u: np.exp(1j * u * (np.log(self.S0) + (self.r - 0.5 * sigma**2) * T)
                                    - 0.5 * sigma**2 * u**2 * T)

        if model == 'heston':
            kappa, eta, theta, rho, V_0 = params
            def char_func(u):
                d = np.sqrt((rho * eta * 1j * u - kappa)**2 + eta**2 * (1j * u + u**2))
                g = (kappa - rho * eta * 1j * u - d) / (kappa - rho * eta * 1j * u + d)
                C = (self.r * 1j * u * T
                     + (kappa * theta / eta**2)
                     * ((kappa - rho * eta * 1j * u - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))))
                D = ((kappa - rho * eta * 1j * u - d) / eta**2
                     * (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T)))
                return np.exp(C + D * V_0 + 1j * u * np.log(self.S0))
            return char_func

        raise NotImplementedError(f"Characteristic function for '{model}' is not yet implemented.")


    def _generate_loss_function(self, model: str, hyper_params: list) -> Callable:
        """
        Build the MSE loss function for a given pricing model.

        Parameters
        ----------
        model : str
            Pricing model name (case-insensitive).
        hyper_params : list
            FFT hyperparameters [N, alpha, eta].

        Returns
        -------
        Callable
            model_func(params) -> float. Computes total squared pricing error
            across all maturities via FFT. Returns 1e10 for invalid parameters.
        """
        _SUPPORTED = {'black-scholes', 'heston', 'bates', 'vg'}

        def is_valid(params):
            if model == 'black-scholes':
                sigma, = params
                return sigma > 0
            if model == 'heston':
                kappa, eta, theta, rho, V_0 = params
                feller = 2 * kappa * theta > eta**2
                return kappa > 0 and eta > 0 and theta > 0 and -1 < rho < 0 and V_0 > 0 and feller
            return True     # no validation for unimplemented models

        model = model.lower()
        if model not in _SUPPORTED:
            raise ValueError(f"Model '{model}' not supported. Choose from: {list(_SUPPORTED)}")

        N, alpha, eta = hyper_params

        def model_func(params):
            if not is_valid(params):
                return 1e10
            total_loss = 0.0
            for i, T in enumerate(self.maturities):
                char_func  = self._build_char_func(model, T, params)
                price_grid = self.fft_pricer(char_func, T, self.strikes[i], N, alpha, eta)
                total_loss += np.sum((price_grid - self.option_prices[i]) ** 2)
            return total_loss

        return model_func
        
    
    def _optimize(self, starting_values: np.ndarray) -> np.ndarray:
        """
        Run Nelder-Mead from multiple starting points and return the best result.

        Parameters
        ----------
        starting_values : np.ndarray
            n × p array of initial parameter guesses.

        Returns
        -------
        optimal_params : np.ndarray
            Parameter vector with the lowest final loss.
        optimize_result : dict
            Contains 'score', 'nit', 'nfev', and 'message' for the best run.
        """
        n, m = starting_values.shape
        
        res_params = np.zeros((n, m))
        res_loss = np.zeros(n)
        meta = []
        
        for idx, init_guess in enumerate(starting_values):
            res = minimize(self.loss_function, init_guess, method="Nelder-Mead")
            
            res_params[idx] = res.x
            res_loss[idx]   = res.fun
            meta.append({
                "message": res.message,
                "nit": res.nit,
                "nfev": res.nfev
                })
            
        res_idx = np.argmin(res_loss)
        
        optimal_params = res_params[res_idx]
        optimize_result = {
            "score": res_loss[res_idx],
            "message": meta[res_idx]["message"],
            "nit": meta[res_idx]["nit"],
            "nfev": meta[res_idx]["nfev"]
            }
        return optimal_params, optimize_result
        
    
    def fft_pricer(self,
         char_func: Callable,
         T: float,
         strikes: np.ndarray,
         N: int = 4096,
         alpha: float = 1.5,
         eta: float = 0.25
         ) -> np.ndarray:
        """
        Price European call options via the Carr-Madan FFT method.

        Parameters
        ----------
        char_func : Callable
            Characteristic function φ(u) of the log-price process.
        T : float
            Time to maturity in years.
        strikes : np.ndarray
            Strike prices at which to return call prices.
        N : int, optional
            Number of FFT grid points (best as power of 2). Default 4096.
        alpha : float, optional
            Damping parameter for the modified call price. Default 1.5.
        eta : float, optional
            Frequency grid spacing. Default 0.25.

        Returns
        -------
        np.ndarray
            Interpolated call prices, one per element of strikes.

        Notes
        -----
        Uses Simpson's rule weights for improved quadrature accuracy.
        """
    
        # --- Grid setup ---
        N = int(N)
        lambda_ = (2 * np.pi / N) / eta
        b = lambda_ * N / 2
    
        j = np.arange(1, N + 1)
        v = eta * (j - 1)
    
        k = -b + lambda_ * (j - 1)  # log-strike grid
    
        # --- Carr-Madan integrand ρ(v) ---
        u = v - (alpha + 1) * 1j
        phi = char_func(u)
        numerator = np.exp(-self.r * T) * phi
        denominator = alpha**2 + alpha - v**2 + 1j * (2*alpha + 1)*v
        rho = numerator / denominator
    
        # --- Simpson's rule weights ---
        sw = (3 + (-1)**j) / 3
        sw[0] = 1 / 3
        sw[-1] = 1 / 3
    
        # --- FFT input vector ---
        fft_input = np.exp(1j * v * b) * rho * eta * sw
    
        # --- Apply FFT and recover call prices ---
        Z = np.real(np.fft.fft(fft_input))
        call_prices_grid = (np.exp(-alpha * k) / np.pi) * Z
    
        # --- Interpolate to requested strikes ---
        log_K = np.log(strikes)
        prices = np.interp(log_K, k, call_prices_grid)
    
        return prices
         
    def plot_performance(
        self,
        estimates: np.ndarray,
        actuals: np.ndarray,
        strikes: np.ndarray,
    ):
        """
        Report and visualise model fit quality.
    
        Parameters
        ----------
        estimates : np.ndarray
            Model prices.
        actuals : np.ndarray
            Market prices.
        strikes : np.ndarray
            Strike prices.
    
        Returns
        -------
        None
            Prints SSE, MSE, RMSE, NRMSE and displays a scatter plot of model vs
            market prices with a residual panel below.
        """
    
        # ------------------------------------------------------------------ #
        #  1. Error metrics                                                    #
        # ------------------------------------------------------------------ #
        def calculate_error(est, act, label):
            n     = len(est)
            SSE   = ((est - act) ** 2).sum()
            MSE   = SSE / n
            RMSE  = MSE ** 0.5
            NRMSE = RMSE / act.mean()
    
            print(f"\n{'='*40}")
            print(f"  {label}")
            print(f"{'='*40}")
            print(f"{'Error Metric':<15} | {'Value':>15}")
            print(f"{'-'*40}")
            print(f"{'SSE':<15} | {SSE:>15.4f}")
            print(f"{'MSE':<15} | {MSE:>15.4f}")
            print(f"{'RMSE':<15} | {RMSE:>15.4f}")
            print(f"{'NRMSE':<15} | {NRMSE:>15.4f}")
            print(f"{'='*40}")
    
        calculate_error(estimates, actuals, "Calibration Set")
    
        # ------------------------------------------------------------------ #
        #  2. Plot config                                                      #
        # ------------------------------------------------------------------ #
        plt.rcParams.update({
            "font.family":       "serif",
            "mathtext.fontset":  "cm",
            "axes.spines.top":   False,
            "axes.spines.right": False,
        })
    
        fig, (ax_main, ax_res) = plt.subplots(
            2, 1,
            figsize=(9, 8),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )
    
        # ------------------------------------------------------------------ #
        #  3. Main panel                                                       #
        # ------------------------------------------------------------------ #
        ax_main.scatter(strikes, actuals,
                        facecolors="none", edgecolors="black",
                        s=55, label="Market Prices")
        ax_main.scatter(strikes, estimates,
                        color="black", marker="+",
                        s=75, label="Model Prices")
    
        ax_main.set_ylabel(r"Option Price $C(K)$")
        ax_main.set_title(rf"Calibration Fit — {self.model}", pad=10)
        ax_main.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
        ax_main.legend(frameon=False)
    
        # ------------------------------------------------------------------ #
        #  4. Residual panel                                                   #
        # ------------------------------------------------------------------ #
        ax_res.scatter(strikes, estimates - actuals,
                       color="black", s=25, alpha=0.7)
    
        ax_res.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax_res.set_xlabel(r"Strike $K$")
        ax_res.set_ylabel(r"Residual $\hat{C} - C$")
        ax_res.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    
        plt.tight_layout()
        plt.show()


    def monte_carlo(self, maturity, steps, contract, contract_params=None, model=None, model_params=None, n_paths=10_000):
        """
        Price one or more derivative contracts via Monte Carlo simulation.
    
        Parameters
        ----------
        maturity : float or np.ndarray
            Time to maturity in years. Can be an array when pricing multiple options.
        steps : int
            Number of time steps in each simulated path.
        contract : str
            One of 'european-call', 'european-put', 'variance-swap', 'variance-swap-fair-strike'.
        contract_params : float or np.ndarray, optional
            Strike price(s) for european contracts; fixed variance strike for 'variance-swap'.
            Must match length of maturity when both are arrays.
        model : str, optional
            Model to simulate under. Defaults to the last calibrated model.
        model_params : np.ndarray, optional
            Model parameters. Defaults to self.optimal_params.
        n_paths : int, optional
            Number of simulated paths. Default is 10_000.
    
        Returns
        -------
        float or np.ndarray
            Single price if both maturity and contract_params are scalar,
            array of prices otherwise.
    
        Notes
        -----
        Paths are simulated once per unique maturity; all strikes for that
        maturity are priced from the same set of paths.
        """
    
        # ------------------------------------------------------------------
        # Path simulators
        # ------------------------------------------------------------------
    
        def _black_scholes(maturity, steps, n_paths, params):
            (sigma,) = params
            dt = maturity / steps
    
            dW = np.random.normal(0, dt ** 0.5, size=(n_paths, steps))
            t  = np.linspace(dt, maturity, steps)
    
            exponent = (self.r - 0.5 * sigma ** 2) * t + sigma * np.cumsum(dW, axis=1)
            paths    = self.S0 * np.exp(exponent)
    
            # prepend S0 column so shape is (n_paths, steps + 1)
            return np.hstack((np.full((n_paths, 1), self.S0), paths))
    
    
        def _heston(maturity, steps, n_paths, params):
            kappa, eta, theta, rho, V0 = params
            dt      = maturity / steps
            sqrt_dt = np.sqrt(dt)
    
            # correlated Brownian increments
            Z1 = np.random.normal(size=(steps, n_paths))
            Z2 = np.random.normal(size=(steps, n_paths))
            dW_v = Z1
            dW_s = rho * Z1 + np.sqrt(1 - rho ** 2) * Z2
    
            vol_paths   = np.empty((steps + 1, n_paths))
            price_paths = np.empty((steps + 1, n_paths))
            vol_paths[0]   = V0
            price_paths[0] = self.S0
    
            for t in range(steps):
                V_t = vol_paths[t]
                vol_paths[t + 1]   = np.abs(V_t + kappa * (theta - V_t) * dt
                                             + eta * np.sqrt(V_t) * dW_v[t] * sqrt_dt)
                price_paths[t + 1] = (price_paths[t]
                                      * np.exp((self.r - 0.5 * V_t) * dt
                                               + np.sqrt(V_t) * dW_s[t] * sqrt_dt))
    
            return price_paths.T  # (n_paths, steps + 1)
    
    
        def _bates(maturity, steps, n_paths, params):
            raise NotImplementedError("Bates model not yet implemented.")
    
    
        def _vg(maturity, steps, n_paths, params):
            raise NotImplementedError("Variance Gamma model not yet implemented.")
    
    
        _MODEL_MAP = {
            'black-scholes': _black_scholes,
            'heston':        _heston,
            'bates':         _bates,
            'vg':            _vg,
        }
    
    
        def _simulate_paths(maturity, steps, n_paths, params, model):
            key = model.lower()
            if key not in _MODEL_MAP:
                raise ValueError(f"Unknown model '{model}'. Choose from {list(_MODEL_MAP)}.")
            return _MODEL_MAP[key](maturity, steps, n_paths, params)
    
    
        # ------------------------------------------------------------------
        # Payoff calculators
        # ------------------------------------------------------------------
    
        def _calc_derivative_value(price_paths, T, strikes):
            """
            price_paths : (n_paths, steps + 1)
            strikes     : 1-D array
            """
            S_T      = price_paths[:, -1]
            discount = np.exp(-self.r * T)
            strikes  = np.asarray(strikes)
            n = price_paths.shape[1] - 1
    
            if contract == 'european-call':
                payoffs = np.maximum(S_T[:, None] - strikes[None, :], 0)
                return discount * payoffs.mean(axis=0)
    
            elif contract == 'european-put':
                payoffs = np.maximum(strikes[None, :] - S_T[:, None], 0)
                return discount * payoffs.mean(axis=0)
    
            elif contract == 'variance-swap':
                log_returns  = np.log(price_paths[:, 1:] / price_paths[:, :-1])
                realised_var = (252 / n) * np.sum(log_returns ** 2, axis=1)  # (n_paths,), annualised
            
                payoffs = realised_var[:, None] - strikes[None, :]  # (n_paths, n_strikes)
                return discount * payoffs.mean(axis=0)
    
            elif contract == 'variance-swap-fair-strike':
                log_returns  = np.log(price_paths[:, 1:] / price_paths[:, :-1])
                realised_var = (252 / n) * np.sum(log_returns ** 2, axis=1)  # annualised
                
                return np.array([realised_var.mean()])
    
            else:
                raise ValueError(f"Unknown contract '{contract}'.")
    
    
        # ------------------------------------------------------------------
        # Main dispatch
        # ------------------------------------------------------------------
    
        model        = model or self.model
        model_params = self.optimal_params if model_params is None else model_params
    
        scalar_input = np.isscalar(maturity) and np.isscalar(contract_params)
        maturity_arr = np.atleast_1d(np.asarray(maturity,        dtype=float))
        strikes_arr  = np.atleast_1d(np.asarray(contract_params, dtype=float))
    
        result = np.empty(len(strikes_arr))
    
        for T in np.unique(maturity_arr):
            mask        = maturity_arr == T
            price_paths = _simulate_paths(T, steps, n_paths, model_params, model)
            result[mask] = _calc_derivative_value(price_paths, T, strikes_arr[mask])
    
        return float(result[0]) if scalar_input else result
        
        
    def vixification(self,
                        T: float,
                        strikes: np.ndarray = None,
                        call_prices: np.ndarray = None,
                        put_prices: np.ndarray = None,
                        N_strikes: int = 500) -> tuple:
        """
        Estimate the fair variance swap strike using the VIX replication formula.

            K_var = (2*e^(rT) / T) * Σ_i [ ΔK_i / K_i² * Q(K_i, T) ] - (1/T) * (F_T/K_0 - 1)²

        where Q is the put price below the forward F_T and the call price at or above it,
        K_0 is the largest strike ≤ F_T, and ΔK_i uses the midpoint rule.

        Parameters
        ----------
        T : float
            Time to maturity in years.
        strikes : np.ndarray, optional
            Raw market strikes. If omitted, a dense grid is generated via FFT.
        call_prices : np.ndarray, optional
            Observed call prices aligned with ``strikes``.
        put_prices : np.ndarray, optional
            Observed put prices aligned with ``strikes``.
        N_strikes : int, optional
            Number of strikes in the synthetic dense grid (model mode only). Default 500.

        Returns
        -------
        K_var : float
            Fair variance swap strike (annualised variance).
        vix : float
            VIX-style estimate: 100 * sqrt(K_var).

        Notes
        -----
        In raw mode, at least one of call_prices or put_prices must be provided;
        the missing leg is derived internally via put-call parity.
        """

        F_T = self.S0 * np.exp((self.r - self.q) * T)

        # ------------------------------------------------------------------
        # Build strike grid and option prices
        # ------------------------------------------------------------------

        raw_mode = strikes is not None

        if raw_mode:
            # --- Raw market mode ------------------------------------------
            strikes = np.asarray(strikes, dtype=float)

            if call_prices is not None and put_prices is not None:
                # Both legs provided directly — use as-is
                calls = np.asarray(call_prices, dtype=float)
                puts  = np.asarray(put_prices,  dtype=float)

            elif call_prices is not None:
                # Derive puts via put-call parity: P = C - S0*e^(-qT) + K*e^(-rT)
                calls = np.asarray(call_prices, dtype=float)
                puts  = calls - self.S0 * np.exp(-self.q * T) + strikes * np.exp(-self.r * T)

            elif put_prices is not None:
                # Derive calls via put-call parity: C = P + S0*e^(-qT) - K*e^(-rT)
                puts  = np.asarray(put_prices, dtype=float)
                calls = puts + self.S0 * np.exp(-self.q * T) - strikes * np.exp(-self.r * T)

            else:
                raise ValueError("Raw mode requires at least one of call_prices or put_prices.")

        else:
            # --- Model-generated mode -------------------------------------
            strikes = np.linspace(0.5 * self.S0, 1.5 * self.S0, N_strikes)
            char_func = self._build_char_func(self.model, T, self.optimal_params)
            calls = self.fft_pricer(char_func, T, strikes)
            puts  = calls - self.S0 * np.exp(-self.q * T) + strikes * np.exp(-self.r * T)

        # ------------------------------------------------------------------
        # Assign Q(K_i): put below forward, call at/above forward
        # ------------------------------------------------------------------

        Q = np.where(strikes < F_T, puts, calls)

        # ------------------------------------------------------------------
        # K_0: largest strike <= F_T
        # ------------------------------------------------------------------

        below = strikes[strikes <= F_T]
        K_0   = below[-1] if len(below) > 0 else strikes[0]

        # ------------------------------------------------------------------
        # Strike spacings via midpoint rule
        # ------------------------------------------------------------------

        dK        = np.empty_like(strikes)
        dK[1:-1]  = (strikes[2:] - strikes[:-2]) / 2   # interior: midpoint
        dK[0]     = strikes[1]  - strikes[0]             # left endpoint
        dK[-1]    = strikes[-1] - strikes[-2]            # right endpoint

        # ------------------------------------------------------------------
        # VIX replication formula
        # ------------------------------------------------------------------

        summation  = np.sum(dK / strikes ** 2 * Q)
        correction = (F_T / K_0 - 1) ** 2

        K_var = (2 * np.exp(self.r * T) / T) * summation - correction / T
        K_var = max(K_var, 0.0)   # guard against small numerical negatives

        vix = 100 * np.sqrt(K_var)

        return K_var, vix


if __name__ == "__main__":

    sns.set_theme(context="notebook", style="whitegrid")

    data = pd.read_csv("Data.csv")
    S0 = 100
    r  = 0.05

    prices     = data["Prices"].to_numpy()
    strikes    = data["Strikes"].to_numpy()
    maturities = data["Maturities"].to_numpy()

    """
    Approach 1: Analytical Approximation
    
        As mentioned during the lectures, the fair strike of a variance swap under Heston
        can be approximated via:
            Kvar ≈ (1 − e**(−κT))*(v0 − η)/κT  + η.
    """
    
    # Initialize with option data
    app = DerivativePricer(
        option_prices = prices,
        strikes       = strikes,
        maturities    = maturities,
        S0 = S0,
        r  = r,
    )

    # To prevent local minima, we optimize (nelder-mead) with different plausible starting values
    starting_values = np.array([
        [1.0, 0.3, 0.05, -0.5, 0.05], # baseline
        [2.0, 0.5, 0.06, -0.7, 0.06], # high mean reversion + strong leverage
        [0.5, 0.2, 0.02, -0.3, 0.02], # low-volatility regime
        [1.5, 0.7, 0.05, -0.6, 0.05], # high vol-of-vol regime
        [1.0, 0.3, 0.08, -0.5, 0.08], # high long-term variance
        [1.0, 0.3, 0.04, -0.9, 0.04], # extreme correlation regime
        [1.0, 0.3, 0.15, -0.5, 0.15], # high initial volatility regime
    ])# columns are respectively (kappa, eta, theta, rho, v_0)

    # Calibrating the model:
    #   1. Define loss function using:
    #       - Relevant characteristique (in the case, Heston)
    #       - Fast Fourier Transform, for ... faster calculations
    #       - MSE to evaluate loss
    #   2. Optimize using Nelder-Mead for different starting points
    app.calibrate_model('heston', starting_values) 
    

    # FFT call prices using calibrated Heston model
    call_prices = np.concatenate([
        app.fft_pricer(app._build_char_func('heston', T, app.optimal_params), T, app.strikes[i])
        for i, T in enumerate(app.maturities)
    ])

    # Plotting performance
    app.plot_performance(
        estimates = call_prices, 
        actuals = prices, 
        strikes = strikes)
        
    # Calculating fair strike for a variance swap (see equation above)
    kappa, eta, theta, rho, V_0 = app.optimal_params
    
    fair_strikes = (1 - np.exp(-kappa * maturities)) * (V_0 - theta) / (kappa * maturities) + theta
    fair_strike_ANAL = fair_strikes[np.isclose(maturities, 1/3)][0]
    
    print(f"\n{'='*40}")
    print(f"{'Parameter':<15} | {'Value':>15}")
    print(f"{'-'*40}")
    print(f"{'kappa':<15} | {kappa:>15.4f}")
    print(f"{'eta':<15} | {eta:>15.4f}")
    print(f"{'theta':<15} | {theta:>15.4f}")
    print(f"{'rho':<15} | {rho:>15.4f}")
    print(f"{'V_0':<15} | {V_0:>15.4f}")
    print(f"{'-'*40}")
    print(f"{'Fair Strike':<15} | {fair_strike_ANAL:>15.4f}")
    print(f"{'='*40}")
    
    
    
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        maturities, fair_strikes, 
        color = 'black', s=50)
    ax.set_xlabel("Maturity")
    ax.set_ylabel("Strike")
    ax.set_title("Fair value of variance swap")
    
    """
    Approach 2: Monte Carlo
    
    """
    
    T = 4/12
    fair_strike_MC = app.monte_carlo(T, round(T*252), contract="variance-swap-fair-strike")[0]
    print(f"Monte Carlo Fair Strike: {fair_strike_MC:.4f}")
    
    
    """
    Approach 3: Vixification

        The fair variance swap strike can be approximated model-free using a
        portfolio of European options across strikes (VIX replication formula).
        This approach is equivalent to the CBOE methodology for computing the VIX.
    """

    T = 4/12
    K_var_VIX, vix_estimate = app.vixification(T)
    print(f"\n{'='*40}")
    print(f"{'Vixification Fair Strike':<25} | {K_var_VIX:>10.4f}")
    print(f"{'VIX Estimate':<25} | {vix_estimate:>10.2f}")
    print(f"{'='*40}")

    # Term structure: VIX across all available maturities
    vix_term_structure = np.array([app.vixification(T)[1] for T in app.maturities])

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(app.maturities, vix_term_structure, color='black', s=50)
    ax.set_xlabel("Maturity")
    ax.set_ylabel("VIX Estimate")
    ax.set_title("VIX Term Structure — Vixification")
    plt.tight_layout()
    plt.show()
