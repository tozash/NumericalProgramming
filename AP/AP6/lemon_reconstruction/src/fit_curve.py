import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.special import comb

def fit_spline(y, r, smoothing_factor=None):
    """
    Fits a smoothing cubic spline to the data (y, r).
    
    Args:
        y, r: Data points.
        smoothing_factor: s parameter for UnivariateSpline.
                          If None, proportional to len(y).
                          
    Returns:
        fitted_r, spline_func: Fitted values at y, and the function object.
    """
    # Sort inputs just to be safe
    idx = np.argsort(y)
    y_sorted = y[idx]
    r_sorted = r[idx]
    
    if smoothing_factor is None:
        # Heuristic: s ~ number of points * variance, or just manual tuning.
        # A large s implies more smoothing.
        smoothing_factor = len(y_sorted) * 10 
    
    spline = UnivariateSpline(y_sorted, r_sorted, k=3, s=smoothing_factor)
    fitted_r = spline(y_sorted)
    
    # Ensure non-negative radius
    fitted_r = np.maximum(fitted_r, 0)
    
    return fitted_r, spline

def fit_bezier(y, r, n_control=4):
    """
    Fits a Bezier curve to the profile (Bonus).
    Since fitting a general Bezier to noisy data is complex, 
    simplification: pick key points (start, end, max width) 
    and manually construct control points or do least squares fit.
    
    We'll implement a simple Least Squares fit for a cubic Bezier (n=3).
    B(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3
    """
    # Normalize y to t in [0, 1]
    y_min, y_max = np.min(y), np.max(y)
    t = (y - y_min) / (y_max - y_min) if y_max > y_min else np.zeros_like(y)
    
    # We want to fit [r(t), y(t)]? Or just r(t)?
    # Usually profile is r vs y. So r = B_r(t), y = B_y(t) with B_y(t) linear?
    # Simple case: r is a function of y.
    
    # Let's compute the Bernstein basis matrix
    # Degree 3
    def bernstein_poly(i, n, t):
        return comb(n, i) * (t**(n-i)) * ((1-t)**i)
        
    n = 3 # Cubic
    A = np.zeros((len(t), n + 1))
    
    for i in range(n + 1):
        A[:, i] = bernstein_poly(i, n, t)
        
    # Solve linear least squares: A * P = r
    # P are the scalar control points for the radius
    P, residuals, rank, s = np.linalg.lstsq(A, r, rcond=None)
    
    fitted_r = A @ P
    fitted_r = np.maximum(fitted_r, 0)
    
    # Create a callable function
    def bezier_func(y_query):
        t_q = (y_query - y_min) / (y_max - y_min)
        # Clip t to [0, 1]
        t_q = np.clip(t_q, 0, 1)
        A_q = np.zeros((len(t_q), n + 1))
        for i in range(n + 1):
            A_q[:, i] = bernstein_poly(i, n, t_q)
        return A_q @ P
        
    return fitted_r, bezier_func
