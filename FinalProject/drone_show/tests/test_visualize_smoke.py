import numpy as np
import pytest
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from drone_show import visualize

def test_visualize_smoke(tmp_path):
    """Test animation generation from dummy trajectory."""
    # Create dummy trajectory: 5 drones, 30 frames, circular motion
    N = 5
    T = 30
    
    # Circular motion: each drone moves in a circle
    times = np.linspace(0, 2*np.pi, T)
    X_series = np.zeros((T, N, 2))
    
    for i in range(N):
        radius = 1.0 + i * 0.2
        phase = i * 2 * np.pi / N
        X_series[:, i, 0] = radius * np.cos(times + phase)
        X_series[:, i, 1] = radius * np.sin(times + phase)
    
    # Save to NPZ
    npz_path = tmp_path / "test_trajectories.npz"
    np.savez_compressed(npz_path, times=times, X=X_series)
    
    # Generate animation
    out_path = tmp_path / "test_animation.gif"
    result_path = visualize.animate_trajectories(
        npz_path, out_path, fps=10, trail=0, show_targets=False
    )
    
    # Verify output exists and is non-empty
    assert Path(result_path).exists(), f"Animation file not created: {result_path}"
    assert Path(result_path).stat().st_size > 0, f"Animation file is empty: {result_path}"
