import numpy as np
import pytest
from drone_show import forces, dynamics

def test_repulsive_forces_zero_when_far():
    """Test that repulsive forces are zero when distance >= Rsafe."""
    Rsafe = 1.0
    k_rep = 1.0
    
    # Two drones far apart
    X = np.array([[0.0, 0.0], [2.0, 0.0]])
    F = forces.repulsive_forces(X, Rsafe, k_rep)
    
    assert np.allclose(F, 0.0), "Forces should be zero when dist > Rsafe"
    
    # Exactly at Rsafe (should be 0 due to < comparison or just formula logic)
    X = np.array([[0.0, 0.0], [1.0, 0.0]])
    F = forces.repulsive_forces(X, Rsafe, k_rep)
    assert np.allclose(F, 0.0), "Forces should be zero when dist == Rsafe"

def test_repulsive_forces_symmetry():
    """Test that two drones exert equal and opposite forces."""
    Rsafe = 2.0
    k_rep = 1.0
    
    # Two drones close to each other
    X = np.array([[0.0, 0.0], [1.0, 0.0]])
    F = forces.repulsive_forces(X, Rsafe, k_rep)
    
    # Force on 0 should be to the left (negative x)
    assert F[0, 0] < 0
    # Force on 1 should be to the right (positive x)
    assert F[1, 0] > 0
    
    # Sum should be zero (Newton's 3rd law)
    assert np.allclose(F[0] + F[1], 0.0)

def test_speed_saturation():
    """Test velocity saturation."""
    vmax = 2.0
    
    # Case 1: Velocity below vmax
    v_slow = np.array([[1.0, 0.0], [0.0, 1.0]])
    v_out = dynamics.speed_saturation(v_slow, vmax)
    assert np.allclose(v_out, v_slow)
    
    # Case 2: Velocity above vmax
    v_fast = np.array([[3.0, 0.0], [0.0, 4.0]])
    v_out = dynamics.speed_saturation(v_fast, vmax)
    
    # Check norms
    norms = np.linalg.norm(v_out, axis=1)
    assert np.allclose(norms, vmax)
    
    # Check direction preservation
    # v_out should be parallel to v_fast
    # normalized vectors should be equal
    n_out = v_out / np.linalg.norm(v_out, axis=1, keepdims=True)
    n_in = v_fast / np.linalg.norm(v_fast, axis=1, keepdims=True)
    assert np.allclose(n_out, n_in)

def test_acceleration_shapes():
    """Test that acceleration returns correct shapes."""
    N = 10
    d = 2
    X = np.zeros((N, d))
    V = np.zeros((N, d))
    T = np.zeros((N, d))
    params = {
        'm': 1.0,
        'kp': 1.0,
        'kd': 0.5,
        'k_rep': 1.0,
        'Rsafe': 0.5,
        'vmax': 2.0
    }
    
    acc = dynamics.acceleration(X, V, T, params)
    assert acc.shape == (N, d)
