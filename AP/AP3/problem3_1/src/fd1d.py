"""
1D finite difference methods for first derivatives.

Implements:
- Central difference (O(h^2))
- Two-point backward difference (O(h^2))
- Richardson extrapolation to boost O(h^2) methods to O(h^4)
"""

import numpy as np
from typing import Callable


def central_diff_1d(f: Callable, x: float, h: float) -> float:
    """
    Compute first derivative using central difference formula (O(h^2)).
    
    Formula: f'(x) ≈ [f(x+h) - f(x-h)] / (2*h)
    
    Args:
        f: Function to differentiate (callable taking single float)
        x: Evaluation point
        h: Step size
        
    Returns:
        Approximate derivative value
    """
    return (f(x + h) - f(x - h)) / (2.0 * h)


def backward2_1d(f: Callable, x: float, h: float) -> float:
    """
    Compute first derivative using two-point backward difference (O(h^2)).
    
    Formula: f'(x) ≈ [3*f(x) - 4*f(x-h) + f(x-2*h)] / (2*h)
    
    Args:
        f: Function to differentiate (callable taking single float)
        x: Evaluation point
        h: Step size
        
    Returns:
        Approximate derivative value
    """
    return (3.0 * f(x) - 4.0 * f(x - h) + f(x - 2.0 * h)) / (2.0 * h)


def richardson_from_order2(D: Callable, f: Callable, x: float, h: float) -> float:
    """
    Apply Richardson extrapolation to boost an O(h^2) derivative method to O(h^4).
    
    Formula: A ≈ [4*D(f, x, h/2) - D(f, x, h)] / 3
    
    This cancels the leading h^2 error term by combining two evaluations
    at different step sizes.
    
    Args:
        D: O(h^2) derivative method (callable: D(f, x, h) -> float)
        f: Function to differentiate
        x: Evaluation point
        h: Step size (will use h and h/2 internally)
        
    Returns:
        Improved derivative estimate (O(h^4))
    """
    D_h = D(f, x, h)
    D_h2 = D(f, x, h / 2.0)
    return (4.0 * D_h2 - D_h) / 3.0

