
import numpy as np
from scipy.interpolate import splprep, splev

def fit_bspline(x_nodes, y_nodes, k=3, s=0, per=False):
    """
    Fits a B-spline to the parametric curve (x(t), y(t)).
    
    Args:
        x_nodes, y_nodes (array): Coordinates.
        k (int): Degree of the spline (3 for cubic, 2 for quadratic).
        s (float): Smoothing factor. 0 = interpolation (goes through all points).
        per (bool): Periodic (closed curve).
        
    Returns:
        tck: Tuple (t, c, k) containing the spline representation.
        u: The parameter values for the knots (0..1 range typically).
    """
    
    # splprep returns (tck, u)
    # tck is the knot vector, coefficients, and degree.
    # u is the parameter values corresponding to the data points.
    
    # Note: nest=-1 is default estimate for number of knots
    tck, u = splprep([x_nodes, y_nodes], k=k, s=s, per=per)
    
    return tck

def eval_bspline(tck, num_points=100):
    """
    Evaluates the B-spline at 'num_points' evenly spaced in [0, 1].
    
    Args:
        tck: The spline tuple from splprep.
        num_points (int): Number of sample points.
        
    Returns:
        x_out, y_out: Arrays of coordinates.
    """
    # Create new parameter space 0..1
    u_new = np.linspace(0, 1, num_points)
    x_out, y_out = splev(u_new, tck)
    return x_out, y_out
