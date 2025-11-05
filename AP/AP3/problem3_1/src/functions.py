"""
Test functions and their exact derivatives for finite difference experiments.

This module defines test functions using SymPy for symbolic computation,
then provides both symbolic and numeric (lambdified) versions.
"""

import numpy as np
import sympy as sp


def get_1d_function(name: str):
    """
    Get a 1D test function and its exact derivative.
    
    Args:
        name: Function name ('f1' for exp(x)*cos(x))
        
    Returns:
        tuple: (f_num, f_prime_num, f_symbolic, f_prime_symbolic)
            - f_num: NumPy callable function
            - f_prime_num: NumPy callable derivative
            - f_symbolic: SymPy expression
            - f_prime_symbolic: SymPy expression for derivative
    """
    x = sp.Symbol('x', real=True)
    
    if name == 'f1':
        # f1(x) = exp(x) * cos(x)
        f_sym = sp.exp(x) * sp.cos(x)
        f_prime_sym = sp.diff(f_sym, x)  # exp(x)*cos(x) - exp(x)*sin(x)
    else:
        raise ValueError(f"Unknown 1D function: {name}")
    
    # Lambdify to NumPy for numeric evaluation
    f_num = sp.lambdify(x, f_sym, 'numpy')
    f_prime_num = sp.lambdify(x, f_prime_sym, 'numpy')
    
    return f_num, f_prime_num, f_sym, f_prime_sym


def get_2d_function(name: str):
    """
    Get a 2D test function and its exact partial derivatives.
    
    Args:
        name: Function name ('g1' for x^2*y + y^2)
        
    Returns:
        tuple: (g_num, gx_num, gy_num, g_symbolic, gx_symbolic, gy_symbolic)
            - g_num: NumPy callable function g(x, y)
            - gx_num: NumPy callable partial derivative g_x(x, y)
            - gy_num: NumPy callable partial derivative g_y(x, y)
            - g_symbolic: SymPy expression
            - gx_symbolic: SymPy expression for g_x
            - gy_symbolic: SymPy expression for g_y
    """
    x, y = sp.symbols('x y', real=True)
    
    if name == 'g1':
        # g1(x, y) = x^2 * y + y^2
        g_sym = x**2 * y + y**2
        gx_sym = sp.diff(g_sym, x)  # 2*x*y
        gy_sym = sp.diff(g_sym, y)  # x^2 + 2*y
    else:
        raise ValueError(f"Unknown 2D function: {name}")
    
    # Lambdify to NumPy for numeric evaluation
    g_num = sp.lambdify((x, y), g_sym, 'numpy')
    gx_num = sp.lambdify((x, y), gx_sym, 'numpy')
    gy_num = sp.lambdify((x, y), gy_sym, 'numpy')
    
    return g_num, gx_num, gy_num, g_sym, gx_sym, gy_sym


def get_2d_function_mixed(name: str):
    """
    Get the mixed partial derivative g_xy for a 2D function.
    
    Args:
        name: Function name
        
    Returns:
        tuple: (gxy_num, gxy_symbolic)
    """
    x, y = sp.symbols('x y', real=True)
    
    if name == 'g1':
        # g1(x, y) = x^2 * y + y^2
        g_sym = x**2 * y + y**2
        gxy_sym = sp.diff(g_sym, x, y)  # 2*x
    else:
        raise ValueError(f"Unknown 2D function: {name}")
    
    gxy_num = sp.lambdify((x, y), gxy_sym, 'numpy')
    
    return gxy_num, gxy_sym

