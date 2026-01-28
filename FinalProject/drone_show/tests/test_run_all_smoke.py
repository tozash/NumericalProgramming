"""
Smoke test for run_all.py pipeline.
"""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from drone_show.tasks import task1, task2, task3
from drone_show import utils, config


def test_run_all_smoke(tmp_path):
    """Test run_all pipeline with tiny N and short times."""
    utils.set_deterministic_behavior(42)
    
    base_output = tmp_path / "run_all_test"
    base_output.mkdir(parents=True, exist_ok=True)
    
    # Task 1: Small scale
    print("Running Task 1...")
    task1_output = base_output / "task1"
    task1_output.mkdir(parents=True, exist_ok=True)
    task1_npz = task1_output / "trajectories.npz"
    
    # Generate text image first
    from drone_show.preprocess import text_to_image
    from PIL import Image
    import numpy as np
    
    img_arr = text_to_image("HI", font_size=30, padding=10, thickness=2)
    img_path = task1_output / "test_text.png"
    img_uint8 = (img_arr * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(img_path)
    
    task1.run_task1(
        image_path=img_path,
        n_drones=20,  # Small N
        duration=2.0,  # Short duration
        dt=0.1,
        output_dir=task1_output,
        seed=42,
        sampling="fill",
        auto_params=True,
        shadow_correct=False
    )
    
    # Assert Task 1 outputs
    assert task1_npz.exists(), "Task 1 NPZ should exist"
    assert (task1_output / "summary.json").exists(), "Task 1 summary should exist"
    assert (task1_output / "animation.mp4").exists(), "Task 1 animation should exist"
    
    # Task 2: Small scale
    print("Running Task 2...")
    task2_output = base_output / "task2"
    task2_npz = task2_output / "trajectories.npz"
    
    task2.run_task2(
        from_npz=task1_npz,
        text="HI",
        duration=2.0,  # Short duration
        dt=0.1,
        output_dir=task2_output,
        seed=42,
        sampling="fill",
        auto_params=True
    )
    
    # Assert Task 2 outputs
    assert task2_npz.exists(), "Task 2 NPZ should exist"
    assert (task2_output / "summary.json").exists(), "Task 2 summary should exist"
    assert (task2_output / "animation.mp4").exists(), "Task 2 animation should exist"
    
    # Task 3: Small scale with synthetic
    print("Running Task 3...")
    task3_output = base_output / "task3"
    
    task3.run_task3(
        from_task2_npz=task2_npz,
        synthetic_video=True,
        dt=0.05,
        T=2.0,  # Short duration
        output_dir=task3_output,
        seed=42,
        auto_params=True
    )
    
    # Assert Task 3 outputs
    assert (task3_output / "trajectories.npz").exists(), "Task 3 NPZ should exist"
    assert (task3_output / "summary.json").exists(), "Task 3 summary should exist"
    assert (task3_output / "animation.mp4").exists(), "Task 3 animation should exist"
    
    # Verify NPZ contents
    import numpy as np
    data1 = np.load(task1_npz)
    assert 'X' in data1 and 'times' in data1
    
    data2 = np.load(task2_npz)
    assert 'X' in data2 and 'times' in data2
    
    data3 = np.load(task3_output / "trajectories.npz")
    assert 'X' in data3 and 'times' in data3
    assert 'centroids_sim' in data3, "Task 3 should have centroids"
    
    print("All pipeline outputs created successfully")


def test_run_all_imports():
    """Test that run_all.py can be imported and main function exists."""
    import importlib.util
    run_all_path = Path(__file__).parent.parent / "scripts" / "run_all.py"
    
    spec = importlib.util.spec_from_file_location("run_all", run_all_path)
    run_all_module = importlib.util.module_from_spec(spec)
    
    # Should not raise
    spec.loader.exec_module(run_all_module)
    
    assert hasattr(run_all_module, 'main'), "run_all.py should have main()"
    assert hasattr(run_all_module, 'run_task1_demo'), "run_all.py should have run_task1_demo()"
    assert hasattr(run_all_module, 'run_task2_demo'), "run_all.py should have run_task2_demo()"
    assert hasattr(run_all_module, 'run_task3_demo'), "run_all.py should have run_task3_demo()"
