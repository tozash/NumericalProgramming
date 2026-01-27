import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from drone_show import targets, utils


def test_task3_targets_preserve_shape():
    """
    Test that rigid translation preserves pairwise distances and centroid matches c(t).
    """
    # Set seed for reproducibility
    utils.set_deterministic_behavior(42)
    
    # Create a synthetic reference formation (random but seeded)
    N = 20
    P_ref = np.random.uniform(-2, 2, (N, 2))
    
    # Compute reference pairwise distances
    from scipy.spatial.distance import pdist
    ref_distances = pdist(P_ref)
    
    # Create a synthetic centroid path (smooth curve)
    t_min, t_max = 0.0, 10.0
    n_samples = 50
    times_sec = np.linspace(t_min, t_max, n_samples)
    
    # Centroid follows a circular path
    theta = np.linspace(0, 2 * np.pi, n_samples)
    radius = 1.5
    centroids_sim = np.column_stack([
        radius * np.cos(theta),
        radius * np.sin(theta)
    ])
    
    # Create centroid interpolator
    c_of_t = targets.make_centroid_interpolator(times_sec, centroids_sim)
    
    # Create rigid translation target function
    target_fn, sample_T_series = targets.make_rigid_translation_targets(P_ref, c_of_t)
    
    # Sample targets at multiple times
    n_test_times = 10
    test_times = np.linspace(t_min, t_max, n_test_times)
    T_series = sample_T_series(test_times)
    
    # Assertions
    tolerance = 1e-8
    
    for i, t in enumerate(test_times):
        T_t = T_series[i]
        
        # 1. Check that pairwise distances are preserved
        T_distances = pdist(T_t)
        max_dist_error = np.max(np.abs(T_distances - ref_distances))
        assert max_dist_error < tolerance, \
            f"Pairwise distances not preserved at t={t:.2f}: max error = {max_dist_error:.2e}"
        
        # 2. Check that centroid of targets matches c(t)
        T_centroid = np.mean(T_t, axis=0)
        c_t = c_of_t(t)
        centroid_error = np.linalg.norm(T_centroid - c_t)
        assert centroid_error < tolerance, \
            f"Centroid mismatch at t={t:.2f}: error = {centroid_error:.2e}"
    
    # Test target_fn directly
    for t in test_times:
        T_t = target_fn(t)
        T_centroid = np.mean(T_t, axis=0)
        c_t = c_of_t(t)
        centroid_error = np.linalg.norm(T_centroid - c_t)
        assert centroid_error < tolerance, \
            f"target_fn centroid mismatch at t={t:.2f}: error = {centroid_error:.2e}"


def test_centroid_interpolator_clamping():
    """Test that centroid interpolator clamps outside time range."""
    times_sec = np.array([0.0, 1.0, 2.0])
    centroids_sim = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0]
    ])
    
    c_of_t = targets.make_centroid_interpolator(times_sec, centroids_sim)
    
    # Test before range (should clamp to first value)
    c_before = c_of_t(-1.0)
    assert np.allclose(c_before, centroids_sim[0]), "Should clamp to first value"
    
    # Test after range (should clamp to last value)
    c_after = c_of_t(5.0)
    assert np.allclose(c_after, centroids_sim[-1]), "Should clamp to last value"
    
    # Test in range (should interpolate)
    c_mid = c_of_t(1.0)
    assert np.allclose(c_mid, centroids_sim[1]), "Should interpolate correctly"


def test_centroid_interpolator_single_point():
    """Test centroid interpolator with single point."""
    times_sec = np.array([0.0])
    centroids_sim = np.array([[1.0, 2.0]])
    
    c_of_t = targets.make_centroid_interpolator(times_sec, centroids_sim)
    
    # Should return constant value
    assert np.allclose(c_of_t(0.0), [1.0, 2.0])
    assert np.allclose(c_of_t(10.0), [1.0, 2.0])
    assert np.allclose(c_of_t(-5.0), [1.0, 2.0])


def test_rigid_translation_preserves_shape():
    """Test that rigid translation preserves the shape exactly."""
    # Create a known shape (square)
    P_ref = np.array([
        [-1, -1],
        [1, -1],
        [1, 1],
        [-1, 1]
    ])
    
    # Constant centroid (no movement)
    times_sec = np.array([0.0, 1.0])
    centroids_sim = np.array([[0.0, 0.0], [0.0, 0.0]])
    
    c_of_t = targets.make_centroid_interpolator(times_sec, centroids_sim)
    target_fn, _ = targets.make_rigid_translation_targets(P_ref, c_of_t)
    
    # At any time, targets should be identical to reference (centroid is at origin)
    T_t = target_fn(0.5)
    assert np.allclose(T_t, P_ref), "Shape should be preserved exactly when centroid doesn't move"
    
    # Now translate centroid
    centroids_sim = np.array([[0.0, 0.0], [2.0, 2.0]])
    c_of_t = targets.make_centroid_interpolator(times_sec, centroids_sim)
    target_fn, _ = targets.make_rigid_translation_targets(P_ref, c_of_t)
    
    # At t=0, should be at origin
    T_0 = target_fn(0.0)
    assert np.allclose(np.mean(T_0, axis=0), [0.0, 0.0])
    
    # At t=1, should be translated by [2, 2]
    T_1 = target_fn(1.0)
    expected_T_1 = P_ref + np.array([2.0, 2.0])
    assert np.allclose(T_1, expected_T_1), "Shape should translate correctly"
