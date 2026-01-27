import numpy as np
import pytest
from pathlib import Path
from drone_show.tasks import task2
from drone_show import utils, config

@pytest.fixture
def synthetic_task1_npz(tmp_path):
    """Creates a synthetic Task 1 NPZ file for testing."""
    npz_path = tmp_path / "task1_synthetic.npz"
    
    N = 60
    T_steps = 50
    
    # Create random seeded positions
    utils.set_deterministic_behavior(42)
    X = np.random.uniform(-3, 3, (T_steps, N, 2))
    V = np.random.uniform(-0.5, 0.5, (T_steps, N, 2))
    times = np.linspace(0, 5.0, T_steps)
    
    # Create some targets (not used in Task 2, but present in NPZ)
    targets = np.random.uniform(-3, 3, (N, 2))
    
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

def test_task2_small(synthetic_task1_npz, tmp_path):
    """Test Task 2 with synthetic Task 1 NPZ."""
    output_dir = tmp_path / "output_task2"
    
    summary = task2.run_task2(
        from_npz=synthetic_task1_npz,
        text="TEST",
        duration=12.0,
        dt=0.02,
        output_dir=output_dir,
        seed=123,
        sampling="fill",
        auto_params=True
    )
    
    # Check outputs exist
    assert (output_dir / "trajectories.npz").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "analysis.json").exists()
    assert (output_dir / "preview.png").exists()
    
    # Check summary stats
    assert summary['n_drones'] == 60
    assert summary['duration'] == 12.0
    
    # Load trajectories
    data = np.load(output_dir / "trajectories.npz", allow_pickle=True)
    X = data['X']
    T_series = data['T_series']
    mean_error = data['mean_error']
    
    assert X.shape[0] == int(12.0 / 0.02) + 1
    assert X.shape == T_series.shape
    
    # Check convergence
    # Note: Initial error might be 0 if drones start exactly at T_start
    # During transition, error may increase then decrease
    # We check that final error is reasonable and that there's improvement from peak
    err_0 = mean_error[0]
    err_f = mean_error[-1]
    err_max = np.max(mean_error)
    
    print(f"Initial Error: {err_0}, Max Error: {err_max}, Final Error: {err_f}")
    
    # Final error should be reasonable
    assert err_f < 0.3, f"Final error too high: {err_f}"
    
    # If error increased during transition, it should have decreased by the end
    if err_max > err_0:
        assert err_f < err_max, f"Error did not decrease from peak: {err_max} -> {err_f}"
    
    # Check collisions
    Rsafe = summary['params']['Rsafe']
    min_dist = summary['final_min_dist']
    print(f"Min dist: {min_dist}, Rsafe: {Rsafe}")
    assert min_dist > 0.5 * Rsafe, f"Collision detected: min_dist={min_dist} <= 0.5*Rsafe={0.5*Rsafe}"
