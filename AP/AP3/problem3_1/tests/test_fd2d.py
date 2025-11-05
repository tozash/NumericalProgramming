"""
Tests for 2D finite difference methods.

Verifies correctness and empirical order of convergence.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from fd2d import central_partial_x, central_partial_y, gradient_central, tangent_plane, normal_vector, mixed_xy_central


def test_central_partial_x_polynomial():
    """Test central partial x on g(x,y) = x^2 * y, g_x = 2*x*y."""
    g = lambda x, y: x**2 * y
    x0, y0 = 2.0, 3.0
    h = 0.1
    exact_gx = 2 * x0 * y0  # 12.0
    
    approx = central_partial_x(g, x0, y0, h)
    error = abs(approx - exact_gx)
    
    assert error < 1e-10, f"Partial x error too large: {error}"


def test_central_partial_y_polynomial():
    """Test central partial y on g(x,y) = x^2 * y, g_y = x^2."""
    g = lambda x, y: x**2 * y
    x0, y0 = 2.0, 3.0
    h = 0.1
    exact_gy = x0**2  # 4.0
    
    approx = central_partial_y(g, x0, y0, h)
    error = abs(approx - exact_gy)
    
    assert error < 1e-10, f"Partial y error too large: {error}"


def test_gradient_central():
    """Test gradient computation returns correct tuple."""
    g = lambda x, y: x**2 * y
    x0, y0 = 2.0, 3.0
    h = 0.1
    
    gx, gy = gradient_central(g, x0, y0, h)
    
    assert isinstance(gx, (float, np.floating))
    assert isinstance(gy, (float, np.floating))
    
    exact_gx = 2 * x0 * y0
    exact_gy = x0**2
    
    assert abs(gx - exact_gx) < 1e-10
    assert abs(gy - exact_gy) < 1e-10


def test_tangent_plane():
    """Test tangent plane construction and evaluation."""
    g = lambda x, y: x**2 * y + y**2
    x0, y0 = 1.0, 3.0
    h = 0.01
    
    # Get exact values
    exact_g0 = g(x0, y0)
    exact_gx = 2 * x0 * y0  # 6.0
    exact_gy = x0**2 + 2 * y0  # 1 + 6 = 7.0
    
    # Build tangent plane
    z_hat = tangent_plane(g, x0, y0, h)
    
    # Evaluate at (x0, y0) - should match g(x0, y0)
    z_at_point = z_hat(x0, y0)
    assert abs(z_at_point - exact_g0) < 1e-6
    
    # Evaluate at nearby point
    z_nearby = z_hat(x0 + 0.1, y0 + 0.1)
    expected = exact_g0 + exact_gx * 0.1 + exact_gy * 0.1
    assert abs(z_nearby - expected) < 1e-4


def test_normal_vector():
    """Test normal vector computation."""
    g = lambda x, y: x**2 * y + y**2
    x0, y0 = 1.0, 3.0
    h = 0.01
    
    n = normal_vector(g, x0, y0, h)
    
    assert isinstance(n, np.ndarray)
    assert n.shape == (3,)
    
    # Exact normal: (-g_x, -g_y, 1)
    exact_gx = 2 * x0 * y0  # 6.0
    exact_gy = x0**2 + 2 * y0  # 7.0
    exact_n = np.array([-exact_gx, -exact_gy, 1.0])
    
    assert np.allclose(n, exact_n, atol=1e-4)


def test_mixed_xy_central():
    """Test mixed partial derivative g_xy."""
    # For g(x,y) = x^2 * y + y^2, g_xy = 2*x
    g = lambda x, y: x**2 * y + y**2
    x0, y0 = 2.0, 3.0
    h = 0.1
    exact_gxy = 2 * x0  # 4.0
    
    approx = mixed_xy_central(g, x0, y0, h)
    error = abs(approx - exact_gxy)
    
    assert error < 1e-8, f"Mixed derivative error too large: {error}"


def test_empirical_order_2d():
    """Empirical order verification for 2D partial derivatives."""
    g = lambda x, y: x**2 * y + y**2
    gx_exact = lambda x, y: 2 * x * y
    gy_exact = lambda x, y: x**2 + 2 * y
    
    x0, y0 = 1.0, 3.0
    exact_gx = gx_exact(x0, y0)
    exact_gy = gy_exact(x0, y0)
    
    h0 = 0.5
    h_values = np.array([h0 / (2**i) for i in range(5)])
    
    # Test g_x
    errors_gx = []
    for h in h_values:
        approx = central_partial_x(g, x0, y0, h)
        errors_gx.append(abs(approx - exact_gx))
    
    errors_gx = np.array(errors_gx)
    from experiment import fit_order_slope
    slope_gx, _ = fit_order_slope(h_values, errors_gx)
    
    # Should be approximately order 2
    assert abs(slope_gx - 2.0) < 0.3, f"g_x order should be ~2, got {slope_gx}"
    
    # Test g_y
    errors_gy = []
    for h in h_values:
        approx = central_partial_y(g, x0, y0, h)
        errors_gy.append(abs(approx - exact_gy))
    
    errors_gy = np.array(errors_gy)
    slope_gy, _ = fit_order_slope(h_values, errors_gy)
    
    # Should be approximately order 2
    assert abs(slope_gy - 2.0) < 0.3, f"g_y order should be ~2, got {slope_gy}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

