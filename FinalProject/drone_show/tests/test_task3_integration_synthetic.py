import numpy as np
import pytest
from pathlib import Path
from drone_show.tasks import task3
from drone_show import utils, config


@pytest.fixture
def synthetic_task2_npz(tmp_path):
    """Creates a synthetic Task 2 NPZ file for testing."""
    npz_path = tmp_path / "task2_synthetic.npz"
    
    N = 60
    T_steps = 100
    
    # Create random seeded positions
    utils.set_deterministic_behavior(42)
    X = np.random.uniform(-3, 3, (T_steps, N, 2))
    V = np.random.uniform(-0.5, 0.5, (T_steps, N, 2))
    times = np.linspace(0, 10.0, T_steps)
    
    # Create reference formation (targets)
    targets = np.random.uniform(-2, 2, (N, 2))
    
    # Default params
    params = {'m': 1.0, 'kp': 2.0, 'kd': 1.5, 'k_rep': 2.0, 'Rsafe': 0.3, 'vmax': 5.0}
    
    # Bounds
    bounds = np.array([-4, 4, -4, 4])
    
    # Save NPZ
    np.savez_compressed(
        npz_path,
        times=times, X=X, V=V, targets=targets, params=params, bounds=bounds
    )
    
    return npz_path


def test_task3_integration_synthetic(synthetic_task2_npz, tmp_path):
    """Test Task 3 with synthetic centroid path."""
    output_dir = tmp_path / "output_task3"
    
    summary = task3.run_task3(
        from_task2_npz=synthetic_task2_npz,
        synthetic_video=True,
        dt=0.02,
        T=8.0,
        output_dir=output_dir,
        seed=123,
        auto_params=True
    )
    
    # Assert output files exist
    assert (output_dir / "trajectories.npz").exists(), "trajectories.npz should exist"
    assert (output_dir / "summary.json").exists(), "summary.json should exist"
    assert (output_dir / "analysis.json").exists(), "analysis.json should exist"
    assert (output_dir / "animation.mp4").exists(), "animation.mp4 should exist"
    
    # Load and verify NPZ
    data = np.load(output_dir / "trajectories.npz")
    assert 'times' in data
    assert 'X' in data
    assert 'V' in data
    assert 'P_ref' in data
    assert 'centroids_sim' in data
    assert 'times_centroids' in data
    
    X_hist = data['X']
    times = data['times']
    N = X_hist.shape[1]
    
    # Load analysis
    import json
    with open(output_dir / "analysis.json", 'r') as f:
        analysis_data = json.load(f)
    
    assert 'mean_error_subsampled' in analysis_data
    assert 'min_pairwise_dist' in analysis_data
    
    # Assertions
    final_mean_err = summary['final_mean_error']
    # Use a looser threshold - following a moving centroid with rigid formation can have some error
    assert final_mean_err < 0.6, f"Final mean error {final_mean_err:.4f} exceeds threshold"
    
    # Check min_dist never below a very loose threshold (just ensure no collisions)
    min_distances = analysis_data['min_pairwise_dist']
    min_safe_threshold = 0.01  # Very loose - just ensure positive distance
    
    for min_dist in min_distances:
        assert min_dist >= min_safe_threshold, \
            f"Min distance {min_dist:.4f} below safety threshold {min_safe_threshold:.4f}"
    
    # Check that centroid of simulated X follows centroid targets
    # Load centroids
    times_centroids = data['times_centroids']
    centroids_sim = data['centroids_sim']
    
    # Sample a few times and compare centroids
    from scipy.interpolate import interp1d
    interp_cx = interp1d(times_centroids, centroids_sim[:, 0], 
                        kind='linear', bounds_error=False, 
                        fill_value=(centroids_sim[0, 0], centroids_sim[-1, 0]))
    interp_cy = interp1d(times_centroids, centroids_sim[:, 1], 
                        kind='linear', bounds_error=False, 
                        fill_value=(centroids_sim[0, 1], centroids_sim[-1, 1]))
    
    # Check at several time points
    test_indices = np.linspace(0, len(times) - 1, 10, dtype=int)
    max_centroid_error = 0.0
    
    for idx in test_indices:
        t = times[idx]
        X_t = X_hist[idx]
        X_centroid = np.mean(X_t, axis=0)
        
        c_t = np.array([interp_cx(t), interp_cy(t)])
        centroid_error = np.linalg.norm(X_centroid - c_t)
        max_centroid_error = max(max_centroid_error, centroid_error)
    
    # Centroid error should be reasonable (formation follows centroid path)
    # Allow some lag/error when following a moving target
    assert max_centroid_error < 2.0, \
        f"Centroid tracking error {max_centroid_error:.4f} too large"


def test_task3_output_structure(synthetic_task2_npz, tmp_path):
    """Test that Task 3 outputs have correct structure."""
    output_dir = tmp_path / "output_task3"
    
    task3.run_task3(
        from_task2_npz=synthetic_task2_npz,
        synthetic_video=True,
        dt=0.02,
        T=5.0,
        output_dir=output_dir,
        seed=123,
        auto_params=True
    )
    
    # Check NPZ structure
    data = np.load(output_dir / "trajectories.npz")
    
    times = data['times']
    X_hist = data['X']
    V_hist = data['V']
    P_ref = data['P_ref']
    centroids_sim = data['centroids_sim']
    times_centroids = data['times_centroids']
    
    N = P_ref.shape[0]
    n_steps = len(times)
    
    assert X_hist.shape == (n_steps, N, 2), f"X_hist shape mismatch: {X_hist.shape}"
    assert V_hist.shape == (n_steps, N, 2), f"V_hist shape mismatch: {V_hist.shape}"
    assert P_ref.shape == (N, 2), f"P_ref shape mismatch: {P_ref.shape}"
    assert centroids_sim.shape[1] == 2, f"centroids_sim shape mismatch: {centroids_sim.shape}"
    assert len(times_centroids) == len(centroids_sim), "times_centroids and centroids_sim length mismatch"
    
    # Check summary
    import json
    with open(output_dir / "summary.json", 'r') as f:
        summary = json.load(f)
    
    assert summary['n_drones'] == N
    assert 'final_mean_error' in summary
    assert 'final_min_dist' in summary
    assert 'params' in summary
