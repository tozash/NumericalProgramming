"""
Generate failure case demonstrations.
"""
import sys
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add src to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from drone_show.tasks import task1, task3
from drone_show import utils, config


def failure_case_1_dt_too_large(output_dir):
    """Failure case 1: dt too large -> unstable / bad convergence."""
    print("Generating Failure Case 1: dt too large")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run with very large dt
    summary = task1.run_task1(
        image_path=None,
        n_drones=60,
        duration=10.0,
        dt=1.0,  # Very large dt
        output_dir=output_dir,
        seed=42,
        sampling="fill",
        auto_params=True,
        shadow_correct=False
    )
    
    final_err = summary['final_mean_error']
    
    # Create README
    readme = f"""Failure Case 1: dt Too Large

Problem:
Using dt=1.0 (very large time step) causes numerical instability and poor convergence.

Results:
- Final Mean Error: {final_err:.4f}
- Expected: < 0.1 for good convergence
- Actual: Much higher due to numerical errors

Explanation:
Large time steps violate the stability requirements of the RK4 integrator.
The simulation becomes unstable and drones cannot converge to targets properly.

Fix:
Use smaller dt (e.g., 0.02-0.1) to ensure stability and accuracy.
"""
    
    with open(output_dir / "README.txt", "w") as f:
        f.write(readme)
    
    print(f"  Saved to {output_dir}")
    return summary


def failure_case_2_Rsafe_too_large(output_dir):
    """Failure case 2: Rsafe too large vs target spacing -> cannot fit letters."""
    print("Generating Failure Case 2: Rsafe too large")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run with manually set large Rsafe
    summary = task1.run_task1(
        image_path=None,
        n_drones=60,
        duration=10.0,
        dt=0.1,
        output_dir=output_dir,
        seed=42,
        sampling="fill",
        auto_params=False,  # Manual params
        shadow_correct=False,
        params={
            'm': 1.0, 'kp': 2.0, 'kd': 1.5, 'k_rep': 2.0,
            'Rsafe': 2.0,  # Very large Rsafe
            'vmax': 5.0
        }
    )
    
    final_err = summary['final_mean_error']
    
    # Create README
    readme = f"""Failure Case 2: Rsafe Too Large

Problem:
Using Rsafe=2.0 (much larger than target spacing) prevents drones from fitting into the formation.

Results:
- Final Mean Error: {final_err:.4f}
- Expected: < 0.1 for good convergence
- Actual: High error because repulsion prevents convergence

Explanation:
When Rsafe is larger than the spacing between target points, the repulsive forces
prevent drones from getting close enough to their targets. The formation cannot
be achieved because drones repel each other even at target positions.

Fix:
Use auto_params=True or set Rsafe < 0.6 * median_target_spacing.
"""
    
    with open(output_dir / "README.txt", "w") as f:
        f.write(readme)
    
    print(f"  Saved to {output_dir}")
    return summary


def failure_case_3_tracking_drift(output_dir):
    """Failure case 3: Tracking drift due to reduced features."""
    print("Generating Failure Case 3: Tracking drift")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a synthetic Task 2 NPZ first
    from drone_show.tasks import task2
    task2_output = output_dir.parent / "task2_for_failure3"
    task2_output.mkdir(parents=True, exist_ok=True)
    
    # Quick Task 1 and Task 2
    task1_output = output_dir.parent / "task1_for_failure3"
    task1_summary = task1.run_task1(
        image_path=None,
        n_drones=60,
        duration=5.0,
        dt=0.1,
        output_dir=task1_output,
        seed=42,
        sampling="fill",
        auto_params=True,
        shadow_correct=False
    )
    
    task2_summary = task2.run_task2(
        from_npz=task1_output / "trajectories.npz",
        text="TEST",
        duration=5.0,
        dt=0.1,
        output_dir=task2_output,
        seed=42,
        sampling="fill",
        auto_params=True
    )
    
    # Now run Task 3 with very low min_features to force drift
    summary = task3.run_task3(
        from_task2_npz=task2_output / "trajectories.npz",
        synthetic_video=True,
        dt=0.02,
        T=6.0,
        output_dir=output_dir,
        seed=42,
        auto_params=True,
        min_features=5  # Very low - would cause drift in real tracking
    )
    
    final_err = summary['final_mean_error']
    
    # Create README
    readme = f"""Failure Case 3: Tracking Drift (Simulated)

Problem:
Using min_features=5 (very low threshold) would cause tracking to fail in real video.

Results:
- Final Mean Error: {final_err:.4f}
- Note: This uses synthetic path, so error is still reasonable
- In real video: Low feature count causes tracking to lose the object

Explanation:
When tracking with optical flow, if the number of tracked features drops too low
(below min_features threshold), the tracker loses accuracy. With min_features=5,
any occlusion or motion blur would cause tracking failure and drift.

Real-world scenario:
- Object moves behind an obstacle
- Lighting changes cause features to disappear
- Fast motion causes blur
- All lead to feature loss and tracking drift

Fix:
- Use higher min_features (default: 30)
- Improve video quality
- Use larger ROI
- Consider alternative tracking methods for challenging scenarios
"""
    
    with open(output_dir / "README.txt", "w") as f:
        f.write(readme)
    
    print(f"  Saved to {output_dir}")
    return summary


def main():
    """Generate all failure cases."""
    output_base = Path("outputs/failures")
    output_base.mkdir(parents=True, exist_ok=True)
    
    utils.set_deterministic_behavior(42)
    
    print("Generating failure case demonstrations...")
    print(f"Output directory: {output_base}\n")
    
    # Case 1: dt too large
    case1_dir = output_base / "case1_dt_too_large"
    summary1 = failure_case_1_dt_too_large(case1_dir)
    
    # Case 2: Rsafe too large
    case2_dir = output_base / "case2_Rsafe_too_large"
    summary2 = failure_case_2_Rsafe_too_large(case2_dir)
    
    # Case 3: Tracking drift
    case3_dir = output_base / "case3_tracking_drift"
    summary3 = failure_case_3_tracking_drift(case3_dir)
    
    print("\n" + "=" * 60)
    print("FAILURE CASES GENERATED")
    print("=" * 60)
    print(f"Case 1 (dt too large): final_error = {summary1['final_mean_error']:.4f}")
    print(f"Case 2 (Rsafe too large): final_error = {summary2['final_mean_error']:.4f}")
    print(f"Case 3 (tracking drift): final_error = {summary3['final_mean_error']:.4f}")
    print(f"\nAll cases saved to: {output_base}")


if __name__ == "__main__":
    main()
