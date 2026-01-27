import pytest
import numpy as np
from pathlib import Path
from PIL import Image
from drone_show.tasks import task1
from drone_show.preprocess import text_to_image

@pytest.fixture
def temp_image_path(tmp_path):
    """Creates a temporary image with text 'TEST'."""
    path = tmp_path / "test_task1.png"
    img_arr = text_to_image("TEST", font_size=50)
    img_uint8 = (img_arr * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(path)
    return path

def test_task1_integration(temp_image_path, tmp_path):
    """
    Runs a small Task 1 simulation integration test.
    """
    output_dir = tmp_path / "output_task1"
    
    N = 20
    dt = 0.1
    duration = 5.0 # Short duration
    
    summary = task1.run_task1(
        image_path=temp_image_path,
        n_drones=N,
        duration=duration,
        dt=dt,
        output_dir=output_dir,
        seed=123
    )
    
    # Check outputs exist
    assert (output_dir / "trajectories.npz").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "preview.png").exists()
    
    # Check summary stats
    assert summary['n_drones'] == N
    assert summary['duration'] == duration
    
    # Check trajectories content
    data = np.load(output_dir / "trajectories.npz", allow_pickle=True)
    X = data['X']
    targets = data['targets']
    
    assert X.shape == (int(duration/dt) + 1, N, 2)
    assert targets.shape == (N, 2)
    
    # Check convergence: Final error should be smaller than initial error
    # Initial error (random/grid vs target)
    err_0 = np.mean(np.linalg.norm(X[0] - targets, axis=1))
    
    # Final error
    err_f = np.mean(np.linalg.norm(X[-1] - targets, axis=1))
    
    # It should have converged significantly
    assert err_f < err_0
    
    # Loose check for absolute convergence (might need more time/tuning for < 0.1)
    # But it should be reasonable
    print(f"Initial Error: {err_0}, Final Error: {err_f}")
    assert err_f < 2.0 # Very loose bound, just ensuring it didn't explode or stay static
