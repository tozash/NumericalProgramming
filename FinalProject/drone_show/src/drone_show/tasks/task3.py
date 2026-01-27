"""
Task 3: Swarm following tracked centroid path with rigid formation.
"""
import numpy as np
import time
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import cv2
from scipy.spatial.distance import pdist
from .. import assignment, dynamics, solver, utils, config, analysis, visualize
from . import task3_targets
from .. import video_tracking, video_io


def run_task3(
    from_task2_npz,
    centroids_csv=None,
    video_path=None,
    synthetic_video=False,
    bbox=None,
    select_roi=False,
    dt=None,
    T=None,
    stride=1,
    output_dir="outputs/task3",
    params=None,
    seed=None,
    auto_params=True,
    max_frames=None,
    min_features=30
):
    """
    Runs Task 3: Swarm simulation following tracked centroid path.
    
    Args:
        from_task2_npz (str or Path): Path to Task 2 trajectories.npz.
        centroids_csv (str or Path, optional): Path to centroids.csv (if already tracked).
        video_path (str or Path, optional): Path to video file for tracking.
        synthetic_video (bool): If True, use synthetic circular centroid path.
        bbox (tuple, optional): Bounding box (x, y, w, h) for tracking.
        select_roi (bool): If True, interactively select ROI.
        dt (float, optional): Time step. Defaults to 1/fps if video provided, else 0.02.
        T (float, optional): Maximum duration. Defaults to video duration or synthetic duration.
        stride (int): Frame stride for tracking.
        output_dir (str or Path): Output directory.
        params (dict, optional): Physics parameters.
        seed (int, optional): Random seed.
        auto_params (bool): Whether to auto-tune parameters.
        max_frames (int, optional): Maximum frames to track.
        min_features (int): Minimum features for tracking.
        
    Returns:
        dict: Summary statistics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if seed is None:
        seed = config.RANDOM_SEED
    utils.set_deterministic_behavior(seed)
    
    print(f"Starting Task 3: Swarm following tracked centroid")
    start_time = time.time()
    
    # 1. Load Task 2 final state
    from_task2_npz = Path(from_task2_npz)
    if not from_task2_npz.exists():
        raise FileNotFoundError(f"Task 2 NPZ not found: {from_task2_npz}")
    
    task2_data = np.load(from_task2_npz)
    X_task2 = task2_data['X']
    V_task2 = task2_data['V']
    times_task2 = task2_data['times']
    
    N = X_task2.shape[1]
    d = 2
    X_start = X_task2[-1].copy()  # Final positions from Task 2
    V_start = np.zeros_like(X_start)  # Start with zero velocity
    
    print(f"Loaded Task 2 final state: N={N} drones")
    
    # 2. Track centroids or use synthetic path
    if centroids_csv is not None:
        # Load pre-tracked centroids
        print(f"Loading centroids from CSV: {centroids_csv}")
        times_sec, centroids_sim = task3_targets.load_centroids_from_csv(centroids_csv)
    elif synthetic_video:
        # Generate synthetic circular path
        print("Using synthetic circular centroid path")
        t_min, t_max = 0.0, 8.0
        n_samples = 100
        times_sec = np.linspace(t_min, t_max, n_samples)
        theta = np.linspace(0, 2 * np.pi, n_samples)
        radius = 1.5
        centroids_sim = np.column_stack([
            radius * np.cos(theta),
            radius * np.sin(theta)
        ])
    elif video_path is not None:
        # Track from video
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        print(f"Tracking centroids from video: {video_path}")
        
        # Get bounding box
        if bbox is None and not select_roi:
            raise ValueError("Must provide --bbox or --select-roi for video tracking")
        
        if select_roi:
            print("Select ROI in the window (press SPACE or ENTER to confirm, ESC to cancel)")
            bbox = video_tracking.select_roi_first_frame(video_path)
            print(f"Selected ROI: {bbox}")
        
        # Track
        times_sec, centroids_px, status_info = video_tracking.track_centroid_optical_flow(
            video_path,
            init_bbox=bbox,
            max_frames=max_frames,
            stride=stride,
            min_features=min_features
        )
        
        # Map to simulation coordinates
        fps, width, height, _ = video_io.read_video_meta(video_path)
        bounds = (-1, 1, -1, 1)  # Default bounds
        centroids_sim = video_tracking.centroids_px_to_sim(centroids_px, width, height, bounds=bounds)
        
        print(f"Tracked {len(times_sec)} centroid samples")
    else:
        raise ValueError("Must provide centroids_csv, video_path, or set synthetic_video=True")
    
    # Determine time parameters
    t_min = times_sec[0]
    t_max = times_sec[-1]
    
    if dt is None:
        if video_path is not None:
            fps, _, _, _ = video_io.read_video_meta(video_path)
            dt = 1.0 / fps
        else:
            dt = 0.02  # Default
    
    # Ensure times start from 0 for simulation (shift if needed)
    if t_min != 0.0:
        times_sec = times_sec - t_min
        t_max = t_max - t_min
        t_min = 0.0
    
    duration = t_max - t_min
    if T is not None:
        duration = min(duration, T)
        t_max = t_min + duration
    
    print(f"Simulation: t=[{t_min:.2f}, {t_max:.2f}], dt={dt:.4f}, duration={duration:.2f}")
    
    # 3. Build target function
    print("Building target function...")
    P_ref = task3_targets.load_reference_formation(from_task2_npz)
    
    # Create centroid interpolator
    from .. import targets
    c_of_t = targets.make_centroid_interpolator(times_sec, centroids_sim)
    
    # Create rigid translation target function
    target_fn, sample_T_series = targets.make_rigid_translation_targets(P_ref, c_of_t)
    
    # 4. Auto-tune parameters
    if auto_params:
        print("Auto-tuning parameters based on reference formation density...")
        params = analysis.auto_params_from_targets(P_ref, base_params=params)
    elif params is None:
        params = {
            'm': 1.0, 'kp': 2.0, 'kd': 1.5, 'k_rep': 2.0, 'Rsafe': 0.3, 'vmax': 5.0
        }
    
    params['total_time'] = duration
    
    print(f"Using parameters: {json.dumps(params, indent=2)}")
    
    # 5. Simulation
    state0 = np.concatenate([X_start.flatten(), V_start.flatten()])
    t_span = (t_min, t_max)
    rhs_func = lambda t, y: dynamics.rhs(t, y, target_fn, params)
    
    print("Simulating...")
    times, states_flat = solver.solve_ivp_rk4(rhs_func, t_span, state0, dt, pbar=True)
    
    # 6. Process results
    n_steps = len(times)
    X_hist = states_flat[:, :N*d].reshape(n_steps, N, d)
    V_hist = states_flat[:, N*d:].reshape(n_steps, N, d)
    
    # Compute target series for error analysis
    T_series = np.zeros((n_steps, N, d))
    for i, t in enumerate(times):
        T_series[i] = target_fn(t)
    
    # Error analysis
    diff = X_hist - T_series
    dist_err = np.linalg.norm(diff, axis=2)
    mean_error = np.mean(dist_err, axis=1)
    max_error = np.max(dist_err, axis=1)
    final_mean_err = mean_error[-1]
    
    # Pairwise distances
    min_pw_dist = []
    for i in range(0, n_steps, max(1, n_steps//100)):
        if N > 1:
            dists = pdist(X_hist[i])
            if len(dists) > 0:
                min_pw_dist.append(np.min(dists))
            else:
                min_pw_dist.append(0.0)
        else:
            min_pw_dist.append(0.0)
    
    final_min_dist = min_pw_dist[-1] if min_pw_dist else 0.0
    
    runtime = time.time() - start_time
    print(f"Simulation finished in {runtime:.2f}s.")
    print(f"Final Mean Error: {final_mean_err:.4f}")
    print(f"Final Min Dist: {final_min_dist:.4f} (Rsafe={params['Rsafe']:.4f})")
    
    # 7. Save outputs
    npz_path = output_dir / "trajectories.npz"
    
    # Store centroids instead of full T_series to save space
    np.savez_compressed(
        npz_path,
        times=times, X=X_hist, V=V_hist, P_ref=P_ref,
        params=params,
        mean_error=mean_error, max_error=max_error,
        times_centroids=times_sec, centroids_sim=centroids_sim
    )
    
    summary = {
        "n_drones": N, "duration": duration, "dt": dt,
        "final_mean_error": float(final_mean_err),
        "final_min_dist": float(final_min_dist),
        "runtime": runtime, "params": params
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Analysis JSON
    times_subsampled = times[::max(1, n_steps//100)]
    analysis_data = {
        "times_subsampled": times_subsampled.tolist(),
        "mean_error_subsampled": mean_error[::max(1, n_steps//100)].tolist(),
        "max_error_subsampled": max_error[::max(1, n_steps//100)].tolist(),
        "min_pairwise_dist": min_pw_dist
    }
    with open(output_dir / "analysis.json", "w") as f:
        json.dump(analysis_data, f)
    
    # 8. Visualization
    print("Generating animation...")
    try:
        animation_path = output_dir / "animation.mp4"
        animate_task3(npz_path, animation_path, target_fn, centroids_sim, times_sec, fps=30)
    except Exception as e:
        print(f"Animation generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"Outputs saved to {output_dir}")
    return summary


def animate_task3(npz_path, output_path, target_fn, centroids_sim, times_centroids, fps=30):
    """
    Creates animation for Task 3 with target overlay and centroid path.
    
    Args:
        npz_path (Path): Path to trajectories.npz.
        output_path (Path): Output path for animation.
        target_fn (callable): Target function target_fn(t) -> (N, 2).
        centroids_sim (np.ndarray): (K, 2) centroid positions.
        times_centroids (np.ndarray): (K,) centroid timestamps.
        fps (int): Frames per second.
    """
    data = np.load(npz_path)
    times = data['times']
    X_hist = data['X']
    P_ref = data['P_ref']
    
    N = X_hist.shape[1]
    
    # Determine bounds
    all_x = np.concatenate([X_hist[:, :, 0].flatten(), P_ref[:, 0], centroids_sim[:, 0]])
    all_y = np.concatenate([X_hist[:, :, 1].flatten(), P_ref[:, 1], centroids_sim[:, 1]])
    x_margin = (np.max(all_x) - np.min(all_x)) * 0.1
    y_margin = (np.max(all_y) - np.min(all_y)) * 0.1
    bounds = (np.min(all_x) - x_margin, np.max(all_x) + x_margin,
              np.min(all_y) - y_margin, np.max(all_y) + y_margin)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # Plot centroid path (static)
    ax.plot(centroids_sim[:, 0], centroids_sim[:, 1], 
           'r-', linewidth=2, alpha=0.5, label='Centroid Path', zorder=1)
    
    # Initialize plots
    drones_scatter = ax.scatter([], [], c='blue', marker='o', s=30, 
                               alpha=0.7, label='Drones', zorder=5)
    targets_scatter = ax.scatter([], [], c='red', marker='x', s=50, 
                                linewidths=2, label='Targets', zorder=6)
    centroid_scatter = ax.scatter([], [], c='green', marker='s', s=100, 
                                 label='Centroid', zorder=7)
    
    ax.legend(loc='upper right')
    
    def animate_frame(i):
        t = times[i]
        X_t = X_hist[i]
        T_t = target_fn(t)
        
        # Find closest centroid time
        centroid_idx = np.argmin(np.abs(times_centroids - t))
        c_t = centroids_sim[centroid_idx]
        
        # Update plots
        drones_scatter.set_offsets(X_t)
        targets_scatter.set_offsets(T_t)
        centroid_scatter.set_offsets([c_t])
        
        ax.set_title(f'Task 3: Swarm Following Centroid Path (t={t:.2f}s)', fontsize=12)
        
        return drones_scatter, targets_scatter, centroid_scatter
    
    # Create animation
    n_frames = len(times)
    interval = 1000 / fps  # milliseconds
    
    anim = FuncAnimation(fig, animate_frame, frames=n_frames, 
                        interval=interval, blit=True, repeat=True)
    
    # Save
    anim.save(str(output_path), writer='ffmpeg', fps=fps, bitrate=1800)
    plt.close()
