import numpy as np
import time
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import cv2
from .. import geometry, assignment, initial_conditions, dynamics, solver, utils, config, analysis, visualize, forces

def run_task2(
    from_npz,
    text="Happy New Year!",
    duration=10.0,
    dt=0.1,
    output_dir="outputs/task2",
    sampling="fill",
    downsample=2,
    bounds=None,
    params=None,
    seed=None,
    auto_params=True,
    T_trans=None
):
    """
    Runs Task 2: Transition from Task 1 final formation to new text formation.
    
    Args:
        from_npz (str or Path): Path to Task 1 trajectories.npz.
        text (str): Target text for new formation.
        duration (float): Simulation duration.
        dt (float): Time step.
        output_dir (str or Path): Output directory.
        sampling (str): "fill" or "edge" for point extraction.
        downsample (int): Downsample factor for fill sampling.
        bounds (tuple, optional): Bounds (xmin, xmax, ymin, ymax). Loaded from NPZ if available.
        params (dict, optional): Physics parameters. Auto-tuned if auto_params=True.
        seed (int, optional): Random seed.
        auto_params (bool): Whether to auto-tune parameters.
        T_trans (float, optional): Transition duration. Defaults to 0.8 * duration.
        
    Returns:
        dict: Summary statistics.
    """
    output_dir = Path(output_dir)
    debug_dir = output_dir / "debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    if seed is None:
        seed = config.RANDOM_SEED
    utils.set_deterministic_behavior(seed)
    
    if T_trans is None:
        T_trans = 0.8 * duration
    
    print(f"Starting Task 2: Transition to '{text}'")
    print(f"Duration={duration}, dt={dt}, T_trans={T_trans}")
    start_time = time.time()
    
    # 1. Load Task 1 NPZ
    from_npz = Path(from_npz)
    if not from_npz.exists():
        raise FileNotFoundError(f"Task 1 NPZ not found: {from_npz}")
    
    data = np.load(from_npz, allow_pickle=True)
    
    X_hist_task1 = data['X']  # Shape: (T, N, 2)
    T_steps_task1, N, d = X_hist_task1.shape
    
    # Extract start positions (final positions from Task 1)
    X_start = X_hist_task1[-1].copy()  # Shape: (N, 2)
    
    # Extract start velocities if available
    if 'V' in data:
        V_hist_task1 = data['V']
        V_start = V_hist_task1[-1].copy()  # Shape: (N, 2)
    else:
        V_start = np.zeros_like(X_start)
    
    # Load bounds from NPZ (fallback to provided or default)
    if 'bounds' in data:
        bounds_array = data['bounds']
        if isinstance(bounds_array, np.ndarray) and len(bounds_array) == 4:
            bounds = tuple(bounds_array)
        else:
            bounds = bounds if bounds is not None else (-4, 4, -4, 4)
    else:
        bounds = bounds if bounds is not None else (-4, 4, -4, 4)
    
    print(f"Loaded Task 1: N={N}, bounds={bounds}")
    
    # 2. Build new target formation
    print(f"Generating target formation from text: '{text}'")
    
    # Generate text image
    img_arr = geometry.preprocess.text_to_image(text, font_size=100, padding=40, thickness=5)
    temp_img_path = debug_dir / "target_text.png"
    img_uint8 = (img_arr * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(temp_img_path)
    
    # Extract points
    def geometry_debug_callback(artifacts):
        if 'mask' in artifacts:
            cv2.imwrite(str(debug_dir / "target_mask.png"), artifacts['mask'] * 255)
    
    target_points_raw = geometry.extract_shape_points_from_image(
        temp_img_path,
        K=N,
        smooth=True,
        debug_callback=geometry_debug_callback,
        sampling=sampling,
        downsample=downsample
    )
    
    # Normalize to same bounds as Task 1
    target_points = assignment.normalize_points(target_points_raw, bounds)
    
    # 3. Assignment
    print("Computing optimal assignment...")
    T_end = assignment.hungarian_assign(X_start, target_points)
    T_start = X_start.copy()  # Start targets = current positions
    
    # 4. Time-varying target function
    def target_fn(t):
        """
        Returns time-varying target positions using smoothstep interpolation.
        """
        tau = np.clip(t / T_trans, 0.0, 1.0)
        s = forces.smoothstep(tau)  # 3*tau^2 - 2*tau^3
        # Interpolate: T(t) = (1-s)*T_start + s*T_end
        T_t = (1.0 - s) * T_start + s * T_end
        return T_t
    
    # 5. Auto-tune parameters
    if auto_params:
        print("Auto-tuning parameters based on final target density...")
        params = analysis.auto_params_from_targets(T_end, base_params=params)
    elif params is None:
        params = {
            'm': 1.0, 'kp': 2.0, 'kd': 1.5, 'k_rep': 2.0, 'Rsafe': 0.3, 'vmax': 5.0
        }
    
    # Inject total_time for repulsion ramping
    params['total_time'] = duration
    
    print(f"Using parameters: {json.dumps(params, indent=2)}")
    
    # 6. Simulation
    state0 = np.concatenate([X_start.flatten(), V_start.flatten()])
    t_span = (0.0, duration)
    rhs_func = lambda t, y: dynamics.rhs(t, y, target_fn, params)
    
    print("Simulating...")
    times, states_flat = solver.solve_ivp_rk4(rhs_func, t_span, state0, dt, pbar=True)
    
    # 7. Process results
    n_steps = len(times)
    X_hist = states_flat[:, :N*d].reshape(n_steps, N, d)
    V_hist = states_flat[:, N*d:].reshape(n_steps, N, d)
    
    # Compute T_series for visualization
    T_series = np.zeros((n_steps, N, d))
    for i, t in enumerate(times):
        T_series[i] = target_fn(t)
    
    # Convergence analysis
    diff = X_hist - T_series
    dist_err = np.linalg.norm(diff, axis=2)
    mean_error = np.mean(dist_err, axis=1)
    max_error = np.max(dist_err, axis=1)
    final_mean_err = mean_error[-1]
    
    # Pairwise distances
    min_pw_dist = []
    from scipy.spatial.distance import pdist
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
    
    # 8. Save outputs
    npz_path = output_dir / "trajectories.npz"
    bounds_array = np.array(bounds)
    np.savez_compressed(
        npz_path,
        times=times, X=X_hist, V=V_hist, T_series=T_series, targets=T_end,
        params=params, bounds=bounds_array,
        mean_error=mean_error, max_error=max_error
    )
    
    summary = {
        "n_drones": N, "duration": duration, "dt": dt, "T_trans": T_trans,
        "final_mean_error": float(final_mean_err),
        "final_min_dist": float(final_min_dist),
        "runtime": runtime, "params": params, "sampling": sampling
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Analysis JSON
    analysis_data = {
        "times_subsampled": times[::max(1, n_steps//100)].tolist(),
        "mean_error_subsampled": mean_error[::max(1, n_steps//100)].tolist(),
        "max_error_subsampled": max_error[::max(1, n_steps//100)].tolist(),
        "min_pairwise_dist": min_pw_dist
    }
    with open(output_dir / "analysis.json", "w") as f:
        json.dump(analysis_data, f)
    
    # 9. Generate Preview
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Determine bounds for plot
    all_x = np.concatenate([X_hist[:, :, 0].flatten(), T_end[:, 0], X_start[:, 0]])
    all_y = np.concatenate([X_hist[:, :, 1].flatten(), T_end[:, 1], X_start[:, 1]])
    x_margin = max((np.max(all_x) - np.min(all_x)) * 0.1, 0.1)
    y_margin = max((np.max(all_y) - np.min(all_y)) * 0.1, 0.1)
    plot_bounds = (np.min(all_x) - x_margin, np.max(all_x) + x_margin,
                   np.min(all_y) - y_margin, np.max(all_y) + y_margin)
    
    title_str = (f"Task 2: Transition Formation (N={N})\n"
                 f"Err={final_mean_err:.3f}, MinDist={final_min_dist:.3f}, Rsafe={params['Rsafe']:.3f}")
    
    # Plot with explicit X_start and X_final
    visualize.plot_frame(ax, X=None, targets=T_end, title=title_str,
                        bounds=plot_bounds, X_start=X_start, X_final=X_hist[-1])
    
    # Add trajectory traces
    for i in range(min(5, N)):
        ax.plot(X_hist[:, i, 0], X_hist[:, i, 1], 'k-', alpha=0.1, linewidth=1)
    
    plt.savefig(output_dir / "preview.png")
    plt.close()
    
    # 10. Generate Animation
    print("Generating animation...")
    try:
        animation_path = output_dir / "animation.mp4"
        visualize.animate_trajectories(npz_path, animation_path, fps=30, trail=0, show_targets=True)
    except Exception as e:
        print(f"Animation generation failed: {e}")
    
    print(f"Outputs saved to {output_dir}")
    return summary
