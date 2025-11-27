# cp1/src/derivatives.py

import numpy as np

"""
Module for numerical differentiation using finite difference methods.
Focus: Taylor-based approximations for 1st-4th derivatives.
"""

def forward_difference(y: np.ndarray, h: float) -> np.ndarray:
    """
    First derivative using Forward Difference: f'(x) = (f(x+h) - f(x)) / h + O(h).
    """
    dy = np.zeros_like(y)
    # Forward difference for all points except the last
    dy[:-1] = (y[1:] - y[:-1]) / h
    # Backward difference for the last point (boundary condition)
    dy[-1] = (y[-1] - y[-2]) / h
    return dy

def central_difference_1st(y: np.ndarray, h: float) -> np.ndarray:
    """
    First derivative using Central Difference: f'(x) = (f(x+h) - f(x-h)) / 2h + O(h^2).
    """
    dy = np.zeros_like(y)
    # Central difference for interior points
    dy[1:-1] = (y[2:] - y[:-2]) / (2 * h)
    
    # Forward difference for first point (O(h)) or 2nd order forward
    # Using 2nd order forward: (-3f(x) + 4f(x+h) - f(x+2h)) / 2h
    if len(y) >= 3:
        dy[0] = (-3*y[0] + 4*y[1] - y[2]) / (2*h)
        dy[-1] = (3*y[-1] - 4*y[-2] + y[-3]) / (2*h) # 2nd order backward
    else:
        # Fallback for very short arrays
        dy[0] = (y[1] - y[0]) / h
        dy[-1] = (y[-1] - y[-2]) / h
        
    return dy

def central_difference_2nd(y: np.ndarray, h: float) -> np.ndarray:
    """
    Second derivative using Central Difference: f''(x) = (f(x+h) - 2f(x) + f(x-h)) / h^2 + O(h^2).
    """
    d2y = np.zeros_like(y)
    # Standard 3-point stencil
    d2y[1:-1] = (y[2:] - 2*y[1:-1] + y[:-2]) / (h**2)
    
    # Boundaries (using forward/backward O(h))
    if len(y) >= 4:
        # Forward 2nd derivative: (2f(x) - 5f(x+1) + 4f(x+2) - f(x+3)) / h^2
        d2y[0] = (2*y[0] - 5*y[1] + 4*y[2] - y[3]) / (h**2)
        d2y[-1] = (2*y[-1] - 5*y[-2] + 4*y[-3] - y[-4]) / (h**2)
        
    return d2y

def central_difference_3rd(y: np.ndarray, h: float) -> np.ndarray:
    """
    Third derivative using Central Difference (4 points involved normally, 
    but standard central for 3rd deriv often uses 2 points before and 2 after).
    Stencil: (-f(x-2h) + 2f(x-h) - 2f(x+h) + f(x+2h)) / (2h^3) + O(h^2).
    """
    d3y = np.zeros_like(y)
    if len(y) < 5:
        return d3y # Not enough points
        
    # Interior points [2:-2]
    d3y[2:-2] = (-y[:-4] + 2*y[1:-3] - 2*y[3:-1] + y[4:]) / (2 * h**3)
    
    # Boundaries are tricky for higher orders, filling with nearest valid or 0
    d3y[0] = d3y[2]
    d3y[1] = d3y[2]
    d3y[-1] = d3y[-3]
    d3y[-2] = d3y[-3]
    
    return d3y

def central_difference_4th(y: np.ndarray, h: float) -> np.ndarray:
    """
    Fourth derivative using Central Difference.
    Stencil: (f(x-2h) - 4f(x-h) + 6f(x) - 4f(x+h) + f(x+2h)) / h^4 + O(h^2).
    """
    d4y = np.zeros_like(y)
    if len(y) < 5:
        return d4y
        
    d4y[2:-2] = (y[:-4] - 4*y[1:-3] + 6*y[2:-2] - 4*y[3:-1] + y[4:]) / (h**4)
    
    # Pad boundaries
    d4y[0:2] = d4y[2]
    d4y[-2:] = d4y[-3]
    
    return d4y

def compute_kinematics(positions: np.ndarray, dt: float):
    """
    Computes velocity, acceleration, jerk, and jounce from position data.
    
    Args:
        positions: Array of shape (N,) or (N, 2) representing position over time.
        dt: Time step.
        
    Returns:
        dict containing arrays for vel, acc, jerk, jounce.
    """
    # Ensure input is float
    pos = positions.astype(float)
    
    # 1. Velocity (1st derivative) - Central O(h^2)
    vel = central_difference_1st(pos, dt)
    
    # 2. Acceleration (2nd derivative) - Central O(h^2)
    acc = central_difference_2nd(pos, dt)
    
    # 3. Jerk (3rd derivative)
    jerk = central_difference_3rd(pos, dt)
    
    # 4. Jounce (4th derivative)
    jounce = central_difference_4th(pos, dt)
    
    return {
        'velocity': vel,
        'acceleration': acc,
        'jerk': jerk,
        'jounce': jounce
    }

