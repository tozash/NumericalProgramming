"""
2D finite difference methods for partial derivatives, gradient, tangent plane, and normal vector.

Implements:
- Central difference for partial derivatives (O(h^2))
- Gradient computation
- Tangent plane construction
- Normal vector computation
- Optional mixed partial derivative (g_xy)
"""

import numpy as np
from typing import Callable


def central_partial_x(g: Callable, x: float, y: float, h: float) -> float:
    """
    Compute partial derivative g_x using central difference (O(h^2)).
    
    Formula: g_x(x,y) ≈ [g(x+h, y) - g(x-h, y)] / (2*h)
    
    Args:
        g: Function g(x, y) to differentiate
        x: x-coordinate of evaluation point
        y: y-coordinate of evaluation point
        h: Step size
        
    Returns:
        Approximate partial derivative g_x
    """
    return (g(x + h, y) - g(x - h, y)) / (2.0 * h)


def central_partial_y(g: Callable, x: float, y: float, h: float) -> float:
    """
    Compute partial derivative g_y using central difference (O(h^2)).
    
    Formula: g_y(x,y) ≈ [g(x, y+h) - g(x, y-h)] / (2*h)
    
    Args:
        g: Function g(x, y) to differentiate
        x: x-coordinate of evaluation point
        y: y-coordinate of evaluation point
        h: Step size
        
    Returns:
        Approximate partial derivative g_y
    """
    return (g(x, y + h) - g(x, y - h)) / (2.0 * h)


def gradient_central(g: Callable, x: float, y: float, h: float) -> tuple:
    """
    Compute gradient (g_x, g_y) using central differences.
    
    Args:
        g: Function g(x, y)
        x: x-coordinate of evaluation point
        y: y-coordinate of evaluation point
        h: Step size
        
    Returns:
        Tuple (gx, gy) of partial derivatives
    """
    gx = central_partial_x(g, x, y, h)
    gy = central_partial_y(g, x, y, h)
    return (gx, gy)


def tangent_plane(g: Callable, x0: float, y0: float, h: float) -> Callable:
    """
    Construct tangent plane to z = g(x, y) at point (x0, y0).
    
    The tangent plane is: z ≈ g(x0, y0) + g_x(x0, y0)*(x - x0) + g_y(x0, y0)*(y - y0)
    
    Args:
        g: Function g(x, y)
        x0: x-coordinate of tangent point
        y0: y-coordinate of tangent point
        h: Step size for computing partial derivatives
        
    Returns:
        Callable function z_hat(x, y) that evaluates the tangent plane
    """
    g0 = g(x0, y0)
    gx, gy = gradient_central(g, x0, y0, h)
    
    def z_hat(x, y):
        """Evaluate tangent plane at (x, y)."""
        return g0 + gx * (x - x0) + gy * (y - y0)
    
    return z_hat


def normal_vector(g: Callable, x0: float, y0: float, h: float) -> np.ndarray:
    """
    Compute normal vector to surface z = g(x, y) at point (x0, y0).
    
    The normal vector is: n = (-g_x, -g_y, 1)
    
    Args:
        g: Function g(x, y)
        x0: x-coordinate of evaluation point
        y0: y-coordinate of evaluation point
        h: Step size for computing partial derivatives
        
    Returns:
        NumPy array [nx, ny, nz] representing the normal vector
    """
    gx, gy = gradient_central(g, x0, y0, h)
    return np.array([-gx, -gy, 1.0])


def mixed_xy_central(g: Callable, x: float, y: float, h: float) -> float:
    """
    Compute mixed partial derivative g_xy using 4-point central stencil (O(h^2)).
    
    Formula: g_xy(x,y) ≈ [g(x+h, y+h) - g(x+h, y-h) - g(x-h, y+h) + g(x-h, y-h)] / (4*h^2)
    
    Args:
        g: Function g(x, y)
        x: x-coordinate of evaluation point
        y: y-coordinate of evaluation point
        h: Step size
        
    Returns:
        Approximate mixed partial derivative g_xy
    """
    return (g(x + h, y + h) - g(x + h, y - h) - g(x - h, y + h) + g(x - h, y - h)) / (4.0 * h**2)

