
import numpy as np
from scipy.interpolate import CubicSpline

def fit_cubic_spline(t_nodes, x_nodes, y_nodes, bc_mode='natural', is_closed=False):
    """
    Fits a parametric cubic spline x(t), y(t).
    
    Args:
        t_nodes (array): Parameter values for the nodes.
        x_nodes, y_nodes (array): Coordinates of the nodes.
        bc_mode (str): 'natural' or 'clamped'.
        is_closed (bool): If True, enforces periodic boundary conditions.
        
    Returns:
        func_x, func_y: Callables that take t and return x, y.
    """
    
    if is_closed:
        # For periodic/closed curves (like 'O'), use bc_type='periodic'
        # ensure first and last points match exactly in data before calling this if needed, 
        # but CubicSpline usually handles it if inputs match.
        cs_x = CubicSpline(t_nodes, x_nodes, bc_type='periodic')
        cs_y = CubicSpline(t_nodes, y_nodes, bc_type='periodic')
        
    elif bc_mode == 'natural':
        # Natural spline: second derivative is zero at endpoints
        cs_x = CubicSpline(t_nodes, x_nodes, bc_type='natural')
        cs_y = CubicSpline(t_nodes, y_nodes, bc_type='natural')
        
    elif bc_mode == 'clamped':
        # Clamped spline: first derivative set to specific values at endpoints.
        # We estimate distinct slopes for x(t) and y(t) using finite differences 
        # of the first two and last two points.
        
        # Slope dx/dt at start ≈ (x[1]-x[0]) / (t[1]-t[0])
        dx_dt_0 = (x_nodes[1] - x_nodes[0]) / (t_nodes[1] - t_nodes[0])
        dx_dt_n = (x_nodes[-1] - x_nodes[-2]) / (t_nodes[-1] - t_nodes[-2])
        
        dy_dt_0 = (y_nodes[1] - y_nodes[0]) / (t_nodes[1] - t_nodes[0])
        dy_dt_n = (y_nodes[-1] - y_nodes[-2]) / (t_nodes[-1] - t_nodes[-2])
        
        cs_x = CubicSpline(t_nodes, x_nodes, bc_type=((1, dx_dt_0), (1, dx_dt_n)))
        cs_y = CubicSpline(t_nodes, y_nodes, bc_type=((1, dy_dt_0), (1, dy_dt_n)))
        
    else:
        raise ValueError(f"Unknown bc_mode: {bc_mode}")
        
    return cs_x, cs_y
