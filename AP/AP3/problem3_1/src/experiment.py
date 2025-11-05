"""
Experiment runner for finite difference accuracy sweeps.

Generates error tables, plots, and fits order of convergence.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

from .functions import get_1d_function, get_2d_function, get_2d_function_mixed
from .fd1d import central_diff_1d, backward2_1d, richardson_from_order2
from .fd2d import central_partial_x, central_partial_y, mixed_xy_central, tangent_plane, normal_vector
from .utils import save_table_to_csv, save_figure


def fit_order_slope(h_values: np.ndarray, errors: np.ndarray) -> Tuple[float, float]:
    """
    Fit the order of convergence by linear regression on log-log data.
    
    Assumes error ≈ C * h^p, so log(error) ≈ log(C) + p * log(h)
    
    Args:
        h_values: Array of step sizes (must be positive)
        errors: Array of absolute errors (must be positive)
        
    Returns:
        Tuple (slope, intercept) where slope is the estimated order p
    """
    # Filter out invalid values (non-positive, NaN, inf)
    mask = (h_values > 0) & (errors > 0) & np.isfinite(h_values) & np.isfinite(errors)
    if np.sum(mask) < 2:
        return np.nan, np.nan
    
    log_h = np.log(h_values[mask])
    log_err = np.log(errors[mask])
    
    # Linear regression: log_err = intercept + slope * log_h
    coeffs = np.polyfit(log_h, log_err, 1)
    slope = coeffs[0]
    intercept = coeffs[1]
    
    return slope, intercept


def run_1d_experiment(func_name: str, x0: float, h0: float, levels: int, 
                     output_dir: str = "paper") -> pd.DataFrame:
    """
    Run 1D finite difference accuracy experiment.
    
    Tests central, backward2, and Richardson extrapolation methods.
    Generates error table and plots.
    
    Args:
        func_name: Name of 1D function ('f1')
        x0: Evaluation point
        h0: Initial step size
        levels: Number of h refinements (geometric: h0, h0/2, h0/4, ...)
        output_dir: Base output directory for tables/figures
        
    Returns:
        DataFrame with columns: method, h, value, exact, abs_error
    """
    # Get function and exact derivative
    f_num, f_prime_num, _, _ = get_1d_function(func_name)
    
    # Generate h grid
    h_values = np.array([h0 / (2**i) for i in range(levels)])
    
    # Compute exact derivative
    exact_prime = f_prime_num(x0)
    
    # Store results
    results = []
    
    # Test each method
    methods = {
        'central': central_diff_1d,
        'backward2': backward2_1d,
    }
    
    for method_name, method_func in methods.items():
        for h in h_values:
            try:
                approx = method_func(f_num, x0, h)
                error = abs(approx - exact_prime)
                results.append({
                    'method': method_name,
                    'h': h,
                    'value': approx,
                    'exact': exact_prime,
                    'abs_error': error
                })
            except Exception as e:
                print(f"Warning: {method_name} failed at h={h}: {e}")
    
    # Test Richardson extrapolation on central difference
    for h in h_values:
        try:
            approx = richardson_from_order2(central_diff_1d, f_num, x0, h)
            error = abs(approx - exact_prime)
            results.append({
                'method': 'richardson(central)',
                'h': h,
                'value': approx,
                'exact': exact_prime,
                'abs_error': error
            })
        except Exception as e:
            print(f"Warning: richardson failed at h={h}: {e}")
    
    df = pd.DataFrame(results)
    
    # Save table
    table_path = f"{output_dir}/tables/1d_errors.csv"
    save_table_to_csv(df, table_path)
    
    # Generate plots
    plot_error_vs_h(df, f"1D Derivative Error vs Step Size (x0={x0})", 
                    f"{output_dir}/figures/1d_error_vs_h.png")
    
    # Print tangent line info
    print(f"\n1D Tangent Line at x0={x0}:")
    print(f"  f(x0) = {f_num(x0):.6f}")
    print(f"  f'(x0) (exact) = {exact_prime:.6f}")
    print(f"  Tangent line: y = {f_num(x0):.6f} + {exact_prime:.6f} * (x - {x0:.6f})")
    
    # Show FD-based tangent line (using smallest h)
    if len(h_values) > 0:
        h_small = h_values[-1]
        fd_prime = central_diff_1d(f_num, x0, h_small)
        print(f"  f'(x0) (FD, h={h_small:.2e}) = {fd_prime:.6f}")
        print(f"  Tangent line (FD): y = {f_num(x0):.6f} + {fd_prime:.6f} * (x - {x0:.6f})")
    
    return df


def run_2d_experiment(func_name: str, x0: float, y0: float, h0: float, levels: int,
                     output_dir: str = "paper", with_mixed: bool = False) -> pd.DataFrame:
    """
    Run 2D finite difference accuracy experiment.
    
    Tests central partial derivatives, optionally mixed derivative.
    Generates error tables and plots.
    
    Args:
        func_name: Name of 2D function ('g1')
        x0: x-coordinate of evaluation point
        y0: y-coordinate of evaluation point
        h0: Initial step size
        levels: Number of h refinements
        output_dir: Base output directory
        with_mixed: Whether to compute mixed derivative g_xy
        
    Returns:
        DataFrame with columns: variable, method, h, value, exact, abs_error
    """
    # Get function and exact partials
    g_num, gx_num, gy_num, _, _, _ = get_2d_function(func_name)
    
    # Generate h grid
    h_values = np.array([h0 / (2**i) for i in range(levels)])
    
    # Compute exact values
    exact_gx = gx_num(x0, y0)
    exact_gy = gy_num(x0, y0)
    
    results = []
    
    # Test g_x
    for h in h_values:
        try:
            approx = central_partial_x(g_num, x0, y0, h)
            error = abs(approx - exact_gx)
            results.append({
                'variable': 'g_x',
                'method': 'central',
                'h': h,
                'value': approx,
                'exact': exact_gx,
                'abs_error': error
            })
        except Exception as e:
            print(f"Warning: g_x failed at h={h}: {e}")
    
    # Test g_y
    for h in h_values:
        try:
            approx = central_partial_y(g_num, x0, y0, h)
            error = abs(approx - exact_gy)
            results.append({
                'variable': 'g_y',
                'method': 'central',
                'h': h,
                'value': approx,
                'exact': exact_gy,
                'abs_error': error
            })
        except Exception as e:
            print(f"Warning: g_y failed at h={h}: {e}")
    
    # Test mixed derivative if requested
    if with_mixed:
        gxy_num, _ = get_2d_function_mixed(func_name)
        exact_gxy = gxy_num(x0, y0)
        for h in h_values:
            try:
                approx = mixed_xy_central(g_num, x0, y0, h)
                error = abs(approx - exact_gxy)
                results.append({
                    'variable': 'g_xy',
                    'method': 'central',
                    'h': h,
                    'value': approx,
                    'exact': exact_gxy,
                    'abs_error': error
                })
            except Exception as e:
                print(f"Warning: g_xy failed at h={h}: {e}")
    
    df = pd.DataFrame(results)
    
    # Save table
    table_path = f"{output_dir}/tables/2d_errors.csv"
    save_table_to_csv(df, table_path)
    
    # Generate plots for each variable
    for var in df['variable'].unique():
        df_var = df[df['variable'] == var]
        plot_error_vs_h(df_var, f"2D Partial Derivative {var} Error vs Step Size (x0={x0}, y0={y0})",
                       f"{output_dir}/figures/2d_{var}_error.png")
    
    # Print tangent plane and normal vector info
    print(f"\n2D Tangent Plane at (x0={x0}, y0={y0}):")
    g0 = g_num(x0, y0)
    print(f"  g(x0, y0) = {g0:.6f}")
    print(f"  g_x(x0, y0) (exact) = {exact_gx:.6f}")
    print(f"  g_y(x0, y0) (exact) = {exact_gy:.6f}")
    print(f"  Tangent plane (exact): z = {g0:.6f} + {exact_gx:.6f}*(x - {x0:.6f}) + {exact_gy:.6f}*(y - {y0:.6f})")
    print(f"  Normal vector (exact): n = ({-exact_gx:.6f}, {-exact_gy:.6f}, 1.0)")
    
    # Show FD-based tangent plane (using smallest h)
    if len(h_values) > 0:
        h_small = h_values[-1]
        fd_gx = central_partial_x(g_num, x0, y0, h_small)
        fd_gy = central_partial_y(g_num, x0, y0, h_small)
        print(f"  g_x(x0, y0) (FD, h={h_small:.2e}) = {fd_gx:.6f}")
        print(f"  g_y(x0, y0) (FD, h={h_small:.2e}) = {fd_gy:.6f}")
        print(f"  Tangent plane (FD): z = {g0:.6f} + {fd_gx:.6f}*(x - {x0:.6f}) + {fd_gy:.6f}*(y - {y0:.6f})")
        print(f"  Normal vector (FD): n = ({-fd_gx:.6f}, {-fd_gy:.6f}, 1.0)")
    
    return df


def plot_error_vs_h(df: pd.DataFrame, title: str, savepath: str) -> None:
    """
    Create log-log plot of error vs step size with fitted slope annotation.
    
    Args:
        df: DataFrame with columns 'h', 'abs_error', and optionally 'method' or 'variable'
        title: Plot title
        savepath: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Determine grouping: by method or by variable
    if 'method' in df.columns:
        group_col = 'method'
    elif 'variable' in df.columns:
        group_col = 'variable'
    else:
        group_col = None
    
    if group_col:
        for group_val in df[group_col].unique():
            df_group = df[df[group_col] == group_val]
            h_vals = df_group['h'].values
            errors = df_group['abs_error'].values
            
            # Filter valid values (allow errors >= 0, but for log-log plot we need errors > 0)
            # Use machine epsilon as minimum for plotting zero errors
            eps = np.finfo(float).eps
            mask = (h_vals > 0) & (errors >= 0) & np.isfinite(h_vals) & np.isfinite(errors)
            h_vals_filtered = h_vals[mask]
            errors_filtered = errors[mask]
            errors_plot = np.maximum(errors_filtered, eps)
            
            if len(h_vals_filtered) > 0:
                ax.loglog(h_vals_filtered, errors_plot, 'o-', label=f'{group_val}', linewidth=2, markersize=6)
                
                # Fit slope and annotate (use original errors for fitting, not plotted ones)
                errors_for_fit = errors_filtered
                # Only fit if we have non-zero errors
                if np.any(errors_for_fit > eps):
                    slope, intercept = fit_order_slope(h_vals_filtered, errors_for_fit)
                    if not np.isnan(slope):
                        ax.text(0.05, 0.95, f'{group_val}: slope ≈ {slope:.2f}',
                               transform=ax.transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                else:
                    # All errors are zero - this is exact!
                    ax.text(0.05, 0.95, f'{group_val}: exact (error = 0)',
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    else:
        # Single curve
        h_vals = df['h'].values
        errors = df['abs_error'].values
        
        # Use machine epsilon as minimum for plotting zero errors
        eps = np.finfo(float).eps
        mask = (h_vals > 0) & (errors >= 0) & np.isfinite(h_vals) & np.isfinite(errors)
        h_vals_filtered = h_vals[mask]
        errors_filtered = errors[mask]
        errors_plot = np.maximum(errors_filtered, eps)
        
        if len(h_vals_filtered) > 0:
            ax.loglog(h_vals_filtered, errors_plot, 'o-', linewidth=2, markersize=6)
            errors_for_fit = errors_filtered
            # Only fit if we have non-zero errors
            if np.any(errors_for_fit > eps):
                slope, intercept = fit_order_slope(h_vals_filtered, errors_for_fit)
                if not np.isnan(slope):
                    ax.text(0.05, 0.95, f'slope ≈ {slope:.2f}',
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            else:
                # All errors are zero - this is exact!
                ax.text(0.05, 0.95, f'exact (error = 0)',
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    ax.set_xlabel('Step size h', fontsize=12)
    ax.set_ylabel('Absolute error', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    if group_col:
        ax.legend()
    
    save_figure(fig, savepath)
    plt.close(fig)


def run_full_experiment(func1d: str, x0_1d: float, h0_1d: float, levels_1d: int,
                       func2d: str, x0_2d: float, y0_2d: float, h0_2d: float, levels_2d: int,
                       output_dir: str = "paper", with_mixed: bool = False) -> Dict:
    """
    Run both 1D and 2D experiments and generate summary.
    
    Returns dictionary with results and fitted slopes.
    """
    print("=" * 60)
    print("Running 1D Experiment")
    print("=" * 60)
    df_1d = run_1d_experiment(func1d, x0_1d, h0_1d, levels_1d, output_dir)
    
    print("\n" + "=" * 60)
    print("Running 2D Experiment")
    print("=" * 60)
    df_2d = run_2d_experiment(func2d, x0_2d, y0_2d, h0_2d, levels_2d, output_dir, with_mixed)
    
    # Compute fitted slopes for summary
    summary = {}
    
    # 1D slopes
    for method in df_1d['method'].unique():
        df_method = df_1d[df_1d['method'] == method]
        h_vals = df_method['h'].values
        errors = df_method['abs_error'].values
        slope, _ = fit_order_slope(h_vals, errors)
        summary[f'1d_{method}_slope'] = slope
        if len(errors) > 0:
            summary[f'1d_{method}_final_error'] = errors[-1]
    
    # 2D slopes
    for var in df_2d['variable'].unique():
        df_var = df_2d[df_2d['variable'] == var]
        h_vals = df_var['h'].values
        errors = df_var['abs_error'].values
        slope, _ = fit_order_slope(h_vals, errors)
        summary[f'2d_{var}_slope'] = slope
        if len(errors) > 0:
            summary[f'2d_{var}_final_error'] = errors[-1]
    
    # Save summary
    summary_path = f"{output_dir}/slopes_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Fitted Order of Convergence (slopes from log-log plots)\n")
        f.write("=" * 60 + "\n\n")
        f.write("1D Methods:\n")
        for key in sorted(summary.keys()):
            if '1d' in key and 'slope' in key:
                method = key.replace('1d_', '').replace('_slope', '')
                f.write(f"  {method}: {summary[key]:.3f}\n")
        f.write("\n2D Methods:\n")
        for key in sorted(summary.keys()):
            if '2d' in key and 'slope' in key:
                var = key.replace('2d_', '').replace('_slope', '')
                f.write(f"  {var}: {summary[key]:.3f}\n")
    
    print(f"\nSaved summary to {summary_path}")
    
    return {'df_1d': df_1d, 'df_2d': df_2d, 'summary': summary}

