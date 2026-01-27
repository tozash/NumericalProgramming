import numpy as np
import pytest
from drone_show import assignment, initial_conditions

def test_initial_positions():
    """Test shape and bounds of initial positions."""
    N = 10
    bounds = (-5, 5, -2, 2)
    
    # Grid
    pos_grid = initial_conditions.initial_positions(N, "grid", bounds)
    assert pos_grid.shape == (N, 2)
    assert np.all(pos_grid[:, 0] >= bounds[0]) and np.all(pos_grid[:, 0] <= bounds[1])
    assert np.all(pos_grid[:, 1] >= bounds[2]) and np.all(pos_grid[:, 1] <= bounds[3])
    
    # Line
    pos_line = initial_conditions.initial_positions(N, "line", bounds)
    assert pos_line.shape == (N, 2)
    assert np.allclose(pos_line[:, 1], 0) # centered in Y ((-2+2)/2 = 0)
    
    # Random
    pos_rand = initial_conditions.initial_positions(N, "random", bounds)
    assert pos_rand.shape == (N, 2)

def test_normalize_points():
    """Test scaling and centering."""
    # Create a small square [0, 1] x [0, 1]
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    
    # Target bounds [-10, 10] x [-10, 10] (size 20x20)
    bounds = (-10, 10, -10, 10)
    
    norm = assignment.normalize_points(points, bounds)
    
    # Should fit max dimension.
    # Aspect ratio is 1. Target aspect is 1.
    # Should scale by 20.
    # New center should be (0,0).
    # Original center was (0.5, 0.5).
    # Expected points: (-10, -10), (10, -10), (10, 10), (-10, 10)
    
    assert np.allclose(np.min(norm, axis=0), -10)
    assert np.allclose(np.max(norm, axis=0), 10)
    
    # Aspect ratio check
    w = np.max(norm[:, 0]) - np.min(norm[:, 0])
    h = np.max(norm[:, 1]) - np.min(norm[:, 1])
    assert np.isclose(w, h)

def test_hungarian_assign():
    """Test optimal assignment."""
    # 3 drones on a line: 0, 1, 2
    X0 = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    
    # 3 targets reversed: 2, 1, 0
    targets = np.array([[2.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    
    # Optimal assignment should map:
    # Drone 0 (0,0) -> Target (0,0) [Index 2]
    # Drone 1 (1,0) -> Target (1,0) [Index 1]
    # Drone 2 (2,0) -> Target (2,0) [Index 0]
    
    assigned = assignment.hungarian_assign(X0, targets)
    
    # assigned[i] is target for drone i.
    # So assigned[0] should be (0,0)
    # assigned[1] should be (1,0)
    # assigned[2] should be (2,0)
    
    expected = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert np.allclose(assigned, expected)
    
    # Check that sum of distances is minimized?
    # Original distance sum (0->2, 1->1, 2->0): 2^2 + 0 + 2^2 = 8
    # Assigned distance sum: 0 + 0 + 0 = 0
    assert np.sum((X0 - assigned)**2) < np.sum((X0 - targets)**2)
