import numpy as np
import pytest
from drone_show import solver

def test_rk4_exponential():
    """
    Test RK4 on y' = y, y(0) = 1. Solution is e^t.
    Check if error scales as O(dt^4).
    """
    rhs = lambda t, y: y
    t_span = (0.0, 1.0)
    y0 = np.array([1.0])
    
    # Run with two different step sizes
    dt1 = 0.1
    t1, y1 = solver.solve_ivp_rk4(rhs, t_span, y0, dt1)
    err1 = np.abs(y1[-1, 0] - np.exp(1.0))
    
    dt2 = 0.05
    t2, y2 = solver.solve_ivp_rk4(rhs, t_span, y0, dt2)
    err2 = np.abs(y2[-1, 0] - np.exp(1.0))
    
    # For 4th order, error should decrease by roughly 2^4 = 16
    ratio = err1 / err2
    
    # Allow some slack: 16 +/- significant margin, but definitely > 2^3 (8)
    # Ideally should be close to 16.
    print(f"Error 1 (dt={dt1}): {err1}")
    print(f"Error 2 (dt={dt2}): {err2}")
    print(f"Ratio: {ratio}")
    
    # Loose bounds to avoid flakiness on different machines/floating point variations
    # Theoretical ratio is ~16.
    assert ratio > 12.0 and ratio < 20.0, f"Convergence order not ~4 (ratio={ratio})"

def test_rk4_harmonic():
    """
    Test RK4 on harmonic oscillator:
    x' = v
    v' = -x
    Exact: x(t) = cos(t), v(t) = -sin(t) (for x0=1, v0=0)
    """
    def rhs(t, state):
        x, v = state
        return np.array([v, -x])
    
    y0 = np.array([1.0, 0.0])
    t_span = (0.0, 2*np.pi)
    dt = 0.01
    
    times, states = solver.solve_ivp_rk4(rhs, t_span, y0, dt)
    
    final_state = states[-1]
    expected_state = np.array([1.0, 0.0]) # Back to start after 2pi
    
    # Check if close to expected
    assert np.allclose(final_state, expected_state, atol=1e-4)

if __name__ == "__main__":
    # verification manual run
    test_rk4_exponential()
    test_rk4_harmonic()
