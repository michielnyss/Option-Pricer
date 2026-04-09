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
        Initialise the pricer with market data and risk-free parameters.

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
         
    
    def plot_performance(self,
                         estimates: np.ndarray, actuals: np.ndarray, strikes: np.ndarray,
                         test_estimates: np.ndarray = None, test_actuals: np.ndarray = None, test_strikes: np.ndarray = None):
        """
        Report and visualise model fit quality, optionally for train and test sets.

        Parameters
        ----------
        estimates : np.ndarray
            Model prices (full dataset or training set).
        actuals : np.ndarray
            Market prices (full dataset or training set).
        strikes : np.ndarray
            Strike prices (full dataset or training set).
        test_estimates : np.ndarray, optional
            Model prices on the test strikes. If None, test set is omitted.
        test_actuals : np.ndarray, optional
            Market prices on the test strikes.
        test_strikes : np.ndarray, optional
            Test strike prices.

        Returns
        -------
        None
            Prints SSE, MSE, RMSE and displays a scatter plot of model vs
            market prices. If test data is provided, both sets are shown.
        """
        def calculate_error(est, act, label):
            n    = len(est)
            SSE  = ((est - act) ** 2).sum()
            MSE  = SSE / n
            RMSE = MSE ** 0.5
            print(f"\n{'='*40}")
            print(f"  {label}")
            print(f"{'='*40}")
            print(f"{'Error Metric':<15} | {'Value':>15}")
            print(f"{'-'*40}")
            print(f"{'SSE':<15} | {SSE:>15.4f}")
            print(f"{'MSE':<15} | {MSE:>15.4f}")
            print(f"{'RMSE':<15} | {RMSE:>15.4f}")
            print(f"{'='*40}")
            return SSE, MSE

        def plot_error():
            fig, ax = plt.subplots(figsize=(9, 6))

            train_label = 'Train' if test_estimates is not None else ''
            ax.scatter(strikes, actuals,   facecolors='none', edgecolors='grey',  s=50, label=f'Market {train_label}'.strip())
            ax.scatter(strikes, estimates, color='grey', marker='+', s=60, zorder=3,    label=f'Model {train_label}'.strip())

            if test_estimates is not None:
                ax.scatter(test_strikes, test_actuals,   facecolors='none', edgecolors='black', s=50,        label='Market Test')
                ax.scatter(test_strikes, test_estimates, color='black', marker='+', s=60, zorder=3,          label='Model Test')

            ax.set_xlabel("Strike")
            ax.set_ylabel("Option Price")
            ax.set_title(f"Model vs Market Prices — {self.model}")
            ax.legend()
            plt.tight_layout()
            plt.show()

        calculate_error(estimates, actuals, "Train" if test_estimates is not None else "All data")
        if test_estimates is not None:
            calculate_error(test_estimates, test_actuals, "Test")
        plot_error()

    def Monte_Carlo(self, maturity, steps, contract, contract_params, model=None, model_params=None, n_paths=10_000):
        """
        Price one or more derivative contracts via Monte Carlo simulation.

        Parameters
        ----------
        maturity : float or np.ndarray
            Time to maturity in years. Can be an array when pricing multiple options.
        steps : int
            Number of time steps in each simulated path.
        contract : str
            Contract type. One of 'european-call', 'european-put', 'variance-swap'.
        contract_params : float or np.ndarray
            Strike price(s). Must be the same length as maturity when both are arrays.
        model : str, optional
            Model to simulate under. Defaults to the last calibrated model.
        model_params : np.ndarray, optional
            Model parameters. Defaults to self.optimal_params.
        n_paths : int, optional
            Number of simulated paths. Default is 10 000.

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
        def simulate_paths(maturity, steps, n_paths, params, model):
            
            def black_scholes(maturity, steps, n_paths, params):
                sigma, = params
                dt = maturity/steps
                
                dW = np.random.normal(0, dt**0.5, size = (n_paths, steps))
                W = np.cumsum(dW, axis = 1) 
                
                t = np.linspace(dt, maturity, steps)
                
                exponent = (self.r - 0.5 * sigma**2) * t + sigma * W
                paths = self.S0 * np.exp(exponent)
                
                paths = np.hstack((self.S0 * np.ones((n_paths, 1)), paths))
                
                return paths
    
            
            def heston(maturity, steps, n_paths, params):
    
                def corr_stand_norm_sampling(rho, steps, n_paths):
                    Z1 = np.random.normal(size=(steps, n_paths))
                    Z2 = np.random.normal(size=(steps, n_paths))
                    W2 = rho * Z1 + (1 - rho**2)**0.5 * Z2
                    return Z1, W2
            
                kappa, eta, theta, rho, V_0 = params
                dt = maturity / steps
                sqrt_dt = np.sqrt(dt)
            
                dW_v, dW_s = corr_stand_norm_sampling(rho, steps, n_paths)
            
                vol_paths   = np.zeros((steps, n_paths))
                price_paths = np.zeros((steps, n_paths))
                vol_paths[0]   = V_0
                price_paths[0] = self.S0
            
                for t in range(steps - 1):
                    V_t = vol_paths[t]
                    volatility = (V_t + kappa * (theta - V_t) * dt
                                  + eta * np.sqrt(V_t) * dW_v[t] * sqrt_dt)
                    vol_paths[t+1] = np.maximum(volatility, -volatility) # because feller condition is not enough :/
                    price_paths[t+1] = (price_paths[t]
                                        * np.exp((self.r - 0.5 * V_t) * dt
                                                 + np.sqrt(V_t) * dW_s[t] * sqrt_dt))
            
                return price_paths.T
            
            def bates(maturity, steps, n_paths, params):
                pass

            def vg(maturity, steps, n_paths, params):
                pass
        
        
            model_map = {
                'black-scholes': black_scholes,
                'heston': heston,
                'bates': bates,
                'vg': vg
                }
            
            simulated_paths = model_map[model.lower()](maturity, steps, n_paths, params)
            
            return simulated_paths
        
        
        def calc_derivative_value(price_paths, T, strikes):
            S_T      = price_paths[:, -1]           # (n_paths,)
            discount = np.exp(-self.r * T)
            strikes  = np.asarray(strikes)          # (n_strikes,)

            if contract == 'european-call':
                payoffs = np.maximum(S_T[:, None] - strikes[None, :], 0)
            elif contract == 'european-put':
                payoffs = np.maximum(strikes[None, :] - S_T[:, None], 0)
            elif contract == 'variance-swap':
                pass  # stub

            return discount * payoffs.mean(axis=0)  # (n_strikes,)


        model        = model or self.model
        model_params = self.optimal_params if model_params is None else model_params

        scalar_input = np.isscalar(maturity) and np.isscalar(contract_params)
        maturity_arr = np.atleast_1d(np.asarray(maturity,         dtype=float))
        strikes_arr  = np.atleast_1d(np.asarray(contract_params,  dtype=float))

        result = np.empty(len(strikes_arr))
        for T in np.unique(maturity_arr):
            mask         = maturity_arr == T
            price_paths  = simulate_paths(T, steps, n_paths, model_params, model)
            result[mask] = calc_derivative_value(price_paths, T, strikes_arr[mask])

        return float(result[0]) if scalar_input else result
        
        
    
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
        [1.0,  0.3,  0.04,  -0.5,  0.04],
        [2.0,  0.5,  0.06,  -0.7,  0.06],
        [0.5,  0.2,  0.02,  -0.3,  0.02],
    ]) # columns are respectively (kappa, eta, theta, rho, v_0)

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
    print(f"\n{'='*40}")
    print(f"{'Parameter':<15} | {'Value':>15}")
    print(f"{'-'*40}")
    print(f"{'kappa':<15} | {kappa:>15.4f}")
    print(f"{'eta':<15} | {eta:>15.4f}")
    print(f"{'theta':<15} | {theta:>15.4f}")
    print(f"{'rho':<15} | {rho:>15.4f}")
    print(f"{'V_0':<15} | {V_0:>15.4f}")
    print(f"{'='*40}")
    
    fair_strikes = (1 - np.exp(-kappa * maturities))*(V_0 - eta) / (kappa * maturities) + eta
    
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
    
        
    