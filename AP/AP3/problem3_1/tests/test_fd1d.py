"""
Tests for 1D finite difference methods.

Verifies correctness and empirical order of convergence.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from fd1d import central_diff_1d, backward2_1d, richardson_from_order2
from experiment import fit_order_slope


def test_central_diff_1d_polynomial():
    """Test central difference on a simple polynomial: f(x) = x^2, f'(x) = 2x."""
    f = lambda x: x**2
    x0 = 2.0
    h = 0.1
    exact = 2 * x0  # 4.0
    
    approx = central_diff_1d(f, x0, h)
    error = abs(approx - exact)
    
    # Should be very accurate for this simple function
    assert error < 1e-10, f"Central difference error too large: {error}"


def test_backward2_1d_polynomial():
    """Test backward difference on a simple polynomial: f(x) = x^2, f'(x) = 2x."""
    f = lambda x: x**2
    x0 = 2.0
    h = 0.1
    exact = 2 * x0  # 4.0
    
    approx = backward2_1d(f, x0, h)
    error = abs(approx - exact)
    
    # Should be very accurate for this simple function
    assert error < 1e-10, f"Backward difference error too large: {error}"


def test_richardson_improves_order():
    """Test that Richardson extrapolation improves order from O(h^2) to O(h^4)."""
    # Use a function with known derivative: f(x) = x^3, f'(x) = 3x^2
    f = lambda x: x**3
    x0 = 1.5
    exact = 3 * x0**2  # 3 * 2.25 = 6.75
    
    # Generate h values
    h0 = 0.1
    h_values = np.array([h0 / (2**i) for i in range(6)])
    
    # Test base method (central) order
    errors_base = []
    for h in h_values:
        approx = central_diff_1d(f, x0, h)
        errors_base.append(abs(approx - exact))
    
    errors_base = np.array(errors_base)
    slope_base, _ = fit_order_slope(h_values, errors_base)
    
    # Test Richardson order
    errors_rich = []
    for h in h_values:
        approx = richardson_from_order2(central_diff_1d, f, x0, h)
        errors_rich.append(abs(approx - exact))
    
    errors_rich = np.array(errors_rich)
    slope_rich, _ = fit_order_slope(h_values, errors_rich)
    
    # Base method should be ~2, Richardson should be ~4
    assert abs(slope_base - 2.0) < 0.3, f"Base method order should be ~2, got {slope_base}"
    assert abs(slope_rich - 4.0) < 0.3, f"Richardson order should be ~4, got {slope_rich}"
    
    # Richardson should have smaller error at smallest h
    assert errors_rich[-1] < errors_base[-1], "Richardson should improve accuracy"


def test_empirical_order_verification():
    """
    Empirical order verification: fit slope of log(error) vs log(h).
    Expect ~2 for base methods, ~4 for Richardson.
    """
    # Use f(x) = exp(x) * cos(x) for a more realistic test
    f = lambda x: np.exp(x) * np.cos(x)
    f_prime = lambda x: np.exp(x) * (np.cos(x) - np.sin(x))
    
    x0 = 0.7
    exact = f_prime(x0)
    
    h0 = 0.5
    h_values = np.array([h0 / (2**i) for i in range(6)])
    
    methods = {
        'central': central_diff_1d,
        'backward2': backward2_1d,
    }
    
    for method_name, method_func in methods.items():
        errors = []
        for h in h_values:
            approx = method_func(f, x0, h)
            errors.append(abs(approx - exact))
        
        errors = np.array(errors)
        slope, _ = fit_order_slope(h_values, errors)
        
        # Should be approximately order 2
        assert abs(slope - 2.0) < 0.3, \
            f"{method_name} order should be ~2, got {slope}"
    
    # Test Richardson
    errors_rich = []
    for h in h_values:
        approx = richardson_from_order2(central_diff_1d, f, x0, h)
        errors_rich.append(abs(approx - exact))
    
    errors_rich = np.array(errors_rich)
    slope_rich, _ = fit_order_slope(h_values, errors_rich)
    
    # Should be approximately order 4
    assert abs(slope_rich - 4.0) < 0.3, \
        f"Richardson order should be ~4, got {slope_rich}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

