import numpy as np
from scipy.stats import norm
from sklearn.utils import Bunch
from typing import Literal, Tuple

Tail = Literal["greater", "less", "two-sided"]

def nonparametric_p_value(estimates, null_estimates, alternative='two-sided',
                          agg_func=np.mean):
    """
    Calculate the nonparametric p-value based on the estimates and null estimates.

    Parameters
    ----------
    estimates : array-like
        Observed estimates (e.g., from data).
    null_estimates : array-like
        Estimates under the null hypothesis (e.g., from permutation or bootstrap).
    alternative : {'two-sided', 'greater', 'less'}
        Type of alternative hypothesis.
    agg_func : callable, default=np.mean
        Function to aggregate the estimates (e.g., mean, median).

    Returns
    -------
    float
        Nonparametric p-value.
    """
    estimates = np.asarray(estimates)
    null_estimates = np.asarray(null_estimates)
    estimate = agg_func(estimates)

    if alternative == 'two-sided':
        centered_null = null_estimates - np.mean(null_estimates)
        position = np.sum(np.abs(centered_null) >= np.abs(estimate - np.mean(null_estimates)))
    elif alternative == 'greater':
        position = np.sum(null_estimates >= estimate)
    elif alternative == 'less':
        position = np.sum(null_estimates <= estimate)
    else:
        raise ValueError("Alternative must be 'two-sided', 'greater', or 'less'.")

    p_value = (position + 1) / (len(null_estimates) + 1)

    if alternative == 'two-sided':
        p_value = min(2 * p_value, 1.0)

    return p_value


def signed_stouffer(pvalues, original_stat):
    """
    author: Jonathan Legrand

    Calculate the signed Stouffer's Z-score and its associated p-value.
    """
    Zi = norm.isf(pvalues / 2) # Divide by 2 for two-tailed test
    k = len(pvalues)
    sign_mask = np.where(original_stat < 0, -1, 1)
    Z = np.sum(Zi * sign_mask) / np.sqrt(k)
    p_global = norm.sf(np.abs(Z)) * 2  # Convert to p-value
    res = Bunch(pvalue=p_global, statistic=Z)
    return res



# Implementation of Extreme Value Theory (EVT) method
# Based on Knijnenburg et al. (2009) "Fewer permutations, more accurate P-values"

import numpy as np
from scipy import stats
from scipy.optimize import minimize

def gpd_fit(data, threshold):
    """
    Fit Generalized Pareto Distribution to data above threshold
    Returns shape (xi) and scale (sigma) parameters
    """
    exceedances = data[data > threshold] - threshold
    n = len(exceedances)
    
    if n == 0:
        return np.nan, np.nan
    
    # Method of moments estimator as initial guess
    mean_exc = np.mean(exceedances)
    var_exc = np.var(exceedances)
    
    if var_exc == 0:
        return 0, mean_exc
    
    # Initial parameter estimates
    xi_init = 0.5 * (mean_exc**2 / var_exc - 1)
    sigma_init = 0.5 * mean_exc * (mean_exc**2 / var_exc + 1)
    
    # Ensure sigma > 0
    if sigma_init <= 0:
        sigma_init = mean_exc
    
    # Maximum likelihood estimation
    def neg_log_likelihood(params):
        xi, sigma = params
        if sigma <= 0:
            return np.inf
        
        if abs(xi) < 1e-6:  # xi ~ 0 case (exponential distribution)
            return n * np.log(sigma) + np.sum(exceedances) / sigma
        else:
            if xi > 0 and np.any(exceedances >= sigma / xi):
                return np.inf
            try:
                log_terms = np.log(1 + xi * exceedances / sigma)
                if np.any(~np.isfinite(log_terms)):
                    return np.inf
                return n * np.log(sigma) + (1 + 1/xi) * np.sum(log_terms)
            except:
                return np.inf
    
    # Optimize
    try:
        result = minimize(lambda x: neg_log_likelihood(x), 
                         [xi_init, sigma_init], 
                         method='Nelder-Mead',
                         options={'maxiter': 1000})
        
        if result.success and result.x[1] > 0:
            return result.x[0], result.x[1]
        else:
            return xi_init, sigma_init
    except:
        return xi_init, sigma_init

def estimate_pvalue_evt(observed_stat, null_stats, alternative='greater', 
                       threshold_quantile=0.95, min_exceedances=10):
    """
    Estimate p-value using Extreme Value Theory as described in 
    Knijnenburg et al. (2009) "Fewer permutations, more accurate P-values"
    
    Parameters:
    -----------
    observed_stat : float
        The observed test statistic
    null_stats : array-like
        Array of null statistics from permutation test
    alternative : str
        'greater', 'less', or 'two-sided'
    threshold_quantile : float
        Quantile to use as threshold for GPD fitting (default 0.95)
    min_exceedances : int
        Minimum number of exceedances required for GPD fitting
        
    Returns:
    --------
    p_value : float
        Estimated p-value using extreme value theory
    method_used : str
        'empirical' or 'evt' indicating which method was used
    """
    
    null_stats = np.array(null_stats)
    n_perms = len(null_stats)
    
    if alternative == 'greater':
        # For right tail test
        threshold = np.quantile(null_stats, threshold_quantile)
        exceedances = null_stats[null_stats > threshold]
        
        if len(exceedances) < min_exceedances or observed_stat <= threshold:
            # Fall back to empirical p-value
            p_emp = np.sum(null_stats >= observed_stat) / n_perms
            return max(p_emp, 1/n_perms), 'empirical'
        
        # Fit GPD to exceedances
        xi, sigma = gpd_fit(null_stats, threshold)
        
        if np.isnan(xi) or np.isnan(sigma) or sigma <= 0:
            p_emp = np.sum(null_stats >= observed_stat) / n_perms
            return max(p_emp, 1/n_perms), 'empirical'
        
        # Calculate p-value using GPD
        n_exceed = len(exceedances)
        prob_exceed_threshold = n_exceed / n_perms
        
        if abs(xi) < 1e-6:  # Exponential case
            p_tail = np.exp(-(observed_stat - threshold) / sigma)
        else:
            if xi > 0 and observed_stat >= threshold + sigma / xi:
                p_tail = 0.0
            else:
                try:
                    p_tail = (1 + xi * (observed_stat - threshold) / sigma) ** (-1/xi)
                except:
                    p_tail = 0.0
        
        p_value = prob_exceed_threshold * p_tail
        
    elif alternative == 'less':
        # For left tail test
        threshold = np.quantile(null_stats, 1 - threshold_quantile)
        exceedances = threshold - null_stats[null_stats < threshold]
        
        if len(exceedances) < min_exceedances or observed_stat >= threshold:
            p_emp = np.sum(null_stats <= observed_stat) / n_perms
            return max(p_emp, 1/n_perms), 'empirical'
        
        # Transform data for GPD fitting
        transformed_null = -null_stats
        transformed_threshold = -threshold
        xi, sigma = gpd_fit(transformed_null, transformed_threshold)
        
        if np.isnan(xi) or np.isnan(sigma) or sigma <= 0:
            p_emp = np.sum(null_stats <= observed_stat) / n_perms
            return max(p_emp, 1/n_perms), 'empirical'
        
        # Calculate p-value using GPD
        n_exceed = len(exceedances)
        prob_exceed_threshold = n_exceed / n_perms
        
        if abs(xi) < 1e-6:  # Exponential case
            p_tail = np.exp(-(threshold - observed_stat) / sigma)
        else:
            if xi > 0 and observed_stat <= threshold - sigma / xi:
                p_tail = 0.0
            else:
                try:
                    p_tail = (1 + xi * (threshold - observed_stat) / sigma) ** (-1/xi)
                except:
                    p_tail = 0.0
        
        p_value = prob_exceed_threshold * p_tail
        
    elif alternative == 'two-sided':
        # For two-sided test, use the minimum of both tails and multiply by 2
        p_greater, method_g = estimate_pvalue_evt(observed_stat, null_stats, 'greater', 
                                                threshold_quantile, min_exceedances)
        p_less, method_l = estimate_pvalue_evt(observed_stat, null_stats, 'less', 
                                             threshold_quantile, min_exceedances)
        
        p_value = 2 * min(p_greater, p_less)
        method_used = 'evt' if 'evt' in [method_g, method_l] else 'empirical'
        
        return min(p_value, 1.0), method_used
    
    else:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'")
    
    return min(max(p_value, 1/n_perms), 1.0), 'evt'

def compare_pvalue_methods(observed_stat, null_stats, alternative='greater'):
    """
    Compare empirical vs EVT p-value estimation
    
    Returns:
    --------
    dict with empirical_p, evt_p, and improvement_factor
    """
    
    null_stats = np.array(null_stats)
    n_perms = len(null_stats)
    
    # Empirical p-value
    if alternative == 'greater':
        p_empirical = np.sum(null_stats >= observed_stat) / n_perms
    elif alternative == 'less':
        p_empirical = np.sum(null_stats <= observed_stat) / n_perms
    elif alternative == 'two-sided':
        p_greater = np.sum(null_stats >= observed_stat) / n_perms
        p_less = np.sum(null_stats <= observed_stat) / n_perms
        p_empirical = 2 * min(p_greater, p_less)
    
    p_empirical = max(p_empirical, 1/n_perms)
    
    # EVT p-value
    p_evt, method = estimate_pvalue_evt(observed_stat, null_stats, alternative)
    
    return {
        'empirical_p': p_empirical,
        'evt_p': p_evt,
        'method_used': method,
        'improvement_factor': p_empirical / p_evt if p_evt > 0 else np.inf,
        'n_permutations': n_perms
    }

def apply_evt_to_analysis(estimates, null_estimates, alternative='greater'):
    """
    Apply EVT method to your existing analysis
    
    Parameters:
    -----------
    estimates : array-like
        Your observed statistics (e.g., correlation scores)
    null_estimates : array-like  
        Your null distribution from permutations
    alternative : str
        'greater', 'less', or 'two-sided'
        
    Returns:
    --------
    results : dict
        Dictionary containing empirical p-values, EVT p-values, and comparison metrics
    """
    
    estimates = np.array(estimates)
    null_estimates = np.array(null_estimates)
    
    results = {
        'empirical_p': [],
        'evt_p': [],
        'method_used': [],
        'improvement_factor': []
    }
    
    # If estimates is a single value, compare against null distribution
    if np.isscalar(estimates) or len(estimates) == 1:
        if np.isscalar(estimates):
            obs_stat = estimates
        else:
            obs_stat = estimates[0]
            
        comparison = compare_pvalue_methods(obs_stat, null_estimates, alternative)
        for key in results.keys():
            results[key] = comparison[key]
    else:
        # If estimates contains multiple values, treat as cross-validation results
        for obs_stat in estimates:
            comparison = compare_pvalue_methods(obs_stat, null_estimates, alternative)
            for key in results.keys():
                results[key].append(comparison[key])
    
    return results


