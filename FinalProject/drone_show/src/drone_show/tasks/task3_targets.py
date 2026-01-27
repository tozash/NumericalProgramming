"""
Task 3 target formation building from tracked centroids.
"""
import numpy as np
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from .. import targets


def load_reference_formation(from_task2_npz):
    """
    Loads reference formation from Task 2 NPZ file.
    
    Prefers stored targets (static) or T_series[-1], falls back to X_series[-1] with warning.
    
    Args:
        from_task2_npz (str or Path): Path to Task 2 trajectories.npz.
        
    Returns:
        np.ndarray: (N, 2) reference formation points P_ref.
    """
    from_task2_npz = Path(from_task2_npz)
    if not from_task2_npz.exists():
        raise FileNotFoundError(f"Task 2 NPZ file not found: {from_task2_npz}")
    
    data = np.load(from_task2_npz)
    
    # Prefer stored targets (static final targets)
    if 'targets' in data:
        P_ref = data['targets']
        print(f"Loaded reference formation from 'targets' ({P_ref.shape})")
    # Fallback to T_series[-1] (time-varying targets at final time)
    elif 'T_series' in data:
        T_series = data['T_series']
        P_ref = T_series[-1]
        print(f"Loaded reference formation from T_series[-1] ({P_ref.shape})")
    # Last resort: use X_series[-1] (actual drone positions, not ideal)
    elif 'X' in data:
        X_series = data['X']
        P_ref = X_series[-1]
        warnings.warn(
            "Using X_series[-1] as reference formation. "
            "This uses actual drone positions, not ideal targets. "
            "Consider using a Task 2 NPZ with 'targets' or 'T_series'.",
            UserWarning
        )
        print(f"Loaded reference formation from X_series[-1] ({P_ref.shape})")
    else:
        raise ValueError(
            "Task 2 NPZ must contain 'targets', 'T_series', or 'X' "
            f"to extract reference formation. Found keys: {list(data.keys())}"
        )
    
    # Ensure shape is (N, 2)
    if len(P_ref.shape) == 3:
        # If it's (1, N, 2), squeeze first dimension
        P_ref = P_ref.reshape(-1, 2)
    elif len(P_ref.shape) != 2 or P_ref.shape[1] != 2:
        raise ValueError(f"Expected reference formation shape (N, 2), got {P_ref.shape}")
    
    return P_ref


def load_centroids_from_csv(centroids_csv):
    """
    Loads tracked centroids from CSV file.
    
    Args:
        centroids_csv (str or Path): Path to centroids.csv.
        
    Returns:
        tuple: (times_sec, centroids_sim)
            - times_sec: (K,) array of timestamps in seconds.
            - centroids_sim: (K, 2) array of centroid positions in simulation coordinates.
    """
    centroids_csv = Path(centroids_csv)
    if not centroids_csv.exists():
        raise FileNotFoundError(f"Centroids CSV file not found: {centroids_csv}")
    
    times_sec = []
    centroids_sim = []
    
    with open(centroids_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            times_sec.append(float(row['time_sec']))
            centroids_sim.append([float(row['x_sim']), float(row['y_sim'])])
    
    times_sec = np.array(times_sec)
    centroids_sim = np.array(centroids_sim)
    
    return times_sec, centroids_sim


def build_task3_targets(from_task2_npz, centroids_csv, bounds=(-1, 1, -1, 1), output_dir=None):
    """
    Builds Task 3 time-varying targets from reference formation and tracked centroids.
    
    Args:
        from_task2_npz (str or Path): Path to Task 2 trajectories.npz.
        centroids_csv (str or Path): Path to centroids.csv from video tracking.
        bounds (tuple): Simulation bounds (xmin, xmax, ymin, ymax) for visualization.
        output_dir (str or Path, optional): Output directory for debug files.
        
    Returns:
        tuple: (target_fn, sample_T_series, P_ref, times_sec, centroids_sim)
            - target_fn: Function target_fn(t) -> (N, 2)
            - sample_T_series: Function sample_T_series(times) -> (K, N, 2)
            - P_ref: (N, 2) reference formation
            - times_sec: (K,) centroid timestamps
            - centroids_sim: (K, 2) centroid positions
    """
    # Load reference formation
    P_ref = load_reference_formation(from_task2_npz)
    N = len(P_ref)
    print(f"Reference formation: {N} drones")
    
    # Load centroids
    times_sec, centroids_sim = load_centroids_from_csv(centroids_csv)
    print(f"Loaded {len(times_sec)} centroid samples")
    
    # Create centroid interpolator
    c_of_t = targets.make_centroid_interpolator(times_sec, centroids_sim)
    
    # Create rigid translation target function
    target_fn, sample_T_series = targets.make_rigid_translation_targets(P_ref, c_of_t)
    
    # Save debug outputs if output_dir provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sample targets at several times for visualization
        t_min = times_sec[0]
        t_max = times_sec[-1]
        n_snapshots = 5
        snapshot_times = np.linspace(t_min, t_max, n_snapshots)
        T_snapshots = sample_T_series(snapshot_times)
        
        # Plot preview
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot reference formation
        ax.scatter(P_ref[:, 0], P_ref[:, 1], c='blue', marker='o', 
                  s=50, label='Reference Formation', alpha=0.7, zorder=3)
        
        # Plot snapshots with different colors
        colors = plt.cm.viridis(np.linspace(0, 1, n_snapshots))
        for i, (t, T_t) in enumerate(zip(snapshot_times, T_snapshots)):
            ax.scatter(T_t[:, 0], T_t[:, 1], c=[colors[i]], marker='x', 
                      s=30, alpha=0.6, zorder=2)
            # Draw lines from reference to snapshot (for first few points)
            if i == 0:
                for j in range(min(5, N)):
                    ax.plot([P_ref[j, 0], T_t[j, 0]], 
                           [P_ref[j, 1], T_t[j, 1]], 
                           'k--', alpha=0.2, linewidth=0.5, zorder=1)
        
        # Plot centroid path
        ax.plot(centroids_sim[:, 0], centroids_sim[:, 1], 
               'r-', linewidth=2, label='Centroid Path', alpha=0.8, zorder=4)
        ax.scatter(centroids_sim[0, 0], centroids_sim[0, 1], 
                  c='green', marker='s', s=100, label='Start', zorder=5)
        ax.scatter(centroids_sim[-1, 0], centroids_sim[-1, 1], 
                  c='red', marker='s', s=100, label='End', zorder=5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Task 3 Target Formation Preview\n(N={N} drones, {len(times_sec)} centroid samples)')
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Set bounds if provided
        if bounds:
            xmin, xmax, ymin, ymax = bounds
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
        
        plt.tight_layout()
        plt.savefig(output_dir / "targets_preview.png", dpi=150)
        plt.close()
        
        # Save T_series for a dense time array
        t_dense = np.linspace(t_min, t_max, 100)
        T_series_dense = sample_T_series(t_dense)
        
        np.savez_compressed(
            output_dir / "T_series.npz",
            times=t_dense,
            T_series=T_series_dense,
            P_ref=P_ref,
            times_centroids=times_sec,
            centroids_sim=centroids_sim
        )
        
        print(f"Debug outputs saved to {output_dir}")
    
    return target_fn, sample_T_series, P_ref, times_sec, centroids_sim
