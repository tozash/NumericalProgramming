"""
Run full pipeline: Task1 -> Task2 -> Task3 (synthetic)
"""
import sys
from pathlib import Path
import json

# Add src to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from drone_show.tasks import task1, task2, task3
from drone_show import utils, config


def run_task1_demo(output_dir, n_drones=100):
    """Run Task 1 with text input."""
    print("=" * 60)
    print("TASK 1: Static Formation")
    print("=" * 60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate text image
    from drone_show.preprocess import text_to_image
    from PIL import Image
    import numpy as np
    
    img_arr = text_to_image("DEMO", font_size=50, padding=20, thickness=3)
    img_path = output_dir / "rendered_text.png"
    img_uint8 = (img_arr * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(img_path)
    
    summary = task1.run_task1(
        image_path=img_path,
        n_drones=n_drones,
        duration=10.0,
        dt=0.1,
        output_dir=output_dir,
        seed=config.RANDOM_SEED,
        sampling="fill",
        auto_params=True,
        shadow_correct=False  # Not needed for text
    )
    
    # Validate outputs
    assert (output_dir / "trajectories.npz").exists(), "Task 1 NPZ missing"
    assert (output_dir / "summary.json").exists(), "Task 1 summary missing"
    assert (output_dir / "animation.mp4").exists(), "Task 1 animation missing"
    
    final_err = summary['final_mean_error']
    print(f"Task 1 Final Mean Error: {final_err:.4f}")
    
    if final_err > 0.5:
        print(f"WARNING: Task 1 error {final_err:.4f} exceeds threshold 0.5")
        return None
    
    return output_dir / "trajectories.npz"


def run_task2_demo(task1_npz, output_dir):
    """Run Task 2 transition."""
    print("\n" + "=" * 60)
    print("TASK 2: Transition Formation")
    print("=" * 60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = task2.run_task2(
        from_npz=task1_npz,
        text="TEST",
        duration=8.0,
        dt=0.1,
        output_dir=output_dir,
        seed=config.RANDOM_SEED,
        sampling="fill",
        auto_params=True
    )
    
    # Validate outputs
    assert (output_dir / "trajectories.npz").exists(), "Task 2 NPZ missing"
    assert (output_dir / "summary.json").exists(), "Task 2 summary missing"
    assert (output_dir / "animation.mp4").exists(), "Task 2 animation missing"
    
    final_err = summary['final_mean_error']
    print(f"Task 2 Final Mean Error: {final_err:.4f}")
    
    if final_err > 0.5:
        print(f"WARNING: Task 2 error {final_err:.4f} exceeds threshold 0.5")
        return None
    
    return output_dir / "trajectories.npz"


def run_task3_demo(task2_npz, output_dir):
    """Run Task 3 with synthetic centroid."""
    print("\n" + "=" * 60)
    print("TASK 3: Swarm Following Centroid")
    print("=" * 60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = task3.run_task3(
        from_task2_npz=task2_npz,
        synthetic_video=True,
        dt=0.02,
        T=6.0,
        output_dir=output_dir,
        seed=config.RANDOM_SEED,
        auto_params=True
    )
    
    # Validate outputs
    assert (output_dir / "trajectories.npz").exists(), "Task 3 NPZ missing"
    assert (output_dir / "summary.json").exists(), "Task 3 summary missing"
    assert (output_dir / "animation.mp4").exists(), "Task 3 animation missing"
    
    final_err = summary['final_mean_error']
    print(f"Task 3 Final Mean Error: {final_err:.4f}")
    
    if final_err > 0.5:
        print(f"WARNING: Task 3 error {final_err:.4f} exceeds threshold 0.5")
        return False
    
    return True


def main():
    """Run full pipeline."""
    utils.set_deterministic_behavior(config.RANDOM_SEED)
    
    base_output = Path("outputs/run_all")
    base_output.mkdir(parents=True, exist_ok=True)
    
    print("Running full pipeline: Task1 -> Task2 -> Task3")
    print(f"Output directory: {base_output}")
    print(f"Seed: {config.RANDOM_SEED}\n")
    
    # Task 1
    task1_npz = run_task1_demo(base_output / "task1", n_drones=100)
    if task1_npz is None:
        print("Task 1 failed - aborting")
        sys.exit(1)
    
    # Task 2
    task2_npz = run_task2_demo(task1_npz, base_output / "task2")
    if task2_npz is None:
        print("Task 2 failed - aborting")
        sys.exit(1)
    
    # Task 3
    success = run_task3_demo(task2_npz, base_output / "task3")
    if not success:
        print("Task 3 failed - aborting")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("ALL TASKS COMPLETED SUCCESSFULLY")
    print("=" * 60)
    
    # Print final metrics
    for task_name in ["task1", "task2", "task3"]:
        summary_path = base_output / task_name / "summary.json"
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            print(f"{task_name.upper()}: final_mean_error = {summary['final_mean_error']:.4f}")


if __name__ == "__main__":
    main()
