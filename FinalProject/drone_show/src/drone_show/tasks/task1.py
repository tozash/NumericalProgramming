import numpy as np
import time
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from .. import geometry, assignment, initial_conditions, dynamics, solver, utils, config, analysis, visualize

def run_task1(
    image_path,
    n_drones,
    duration,
    dt,
    output_dir,
    params=None,
    seed=None,
    sampling="fill",
    downsample=2,
    auto_params=True,
    shadow_correct=False,
    shadow_k_frac=0.12,
    shadow_method="divide",
    thresh_mode="adaptive",
    thresh_block_size=35,
    thresh_C=10,
    edge_from_mask="morph",
    canny_low=50,
    canny_high=150
):
    """
    Runs Task 1: Static Formation on Handwritten Input.
    """
    output_dir = Path(output_dir)
    debug_dir = output_dir / "debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    if seed is None:
        seed = config.RANDOM_SEED
    utils.set_deterministic_behavior(seed)
    
    print(f"Starting Task 1 with N={n_drones}, T={duration}, dt={dt}, mode={sampling}, auto_params={auto_params}")
    start_time = time.time()
    
    # 1. Pipeline
    def geometry_debug_callback(artifacts):
        # Save debug images in order: 00_gray.png, 01_corr.png, 02_mask.png, 03_edges.png
        if 'gray' in artifacts:
            cv2.imwrite(str(debug_dir / "00_gray.png"), artifacts['gray'])
        if 'corr' in artifacts:
            cv2.imwrite(str(debug_dir / "01_corr.png"), artifacts['corr'])
        if 'mask' in artifacts:
            # Handle both uint8 {0,255} and float {0,1} masks
            mask = artifacts['mask']
            if mask.dtype == np.float32 or mask.dtype == np.float64:
                mask = (mask * 255).astype(np.uint8)
            cv2.imwrite(str(debug_dir / "02_mask.png"), mask)
        if 'edges' in artifacts:
            cv2.imwrite(str(debug_dir / "03_edges.png"), artifacts['edges'])
        if 'contours' in artifacts:
            # Contours are already saved via edges, but we can visualize them
            pass

    try:
        target_points_raw = geometry.extract_shape_points_from_image(
            image_path, 
            K=n_drones, 
            smooth=True,
            debug_callback=geometry_debug_callback,
            sampling=sampling,
            downsample=downsample,
            shadow_correct=shadow_correct,
            shadow_k_frac=shadow_k_frac,
            shadow_method=shadow_method,
            thresh_mode=thresh_mode,
            thresh_block_size=thresh_block_size,
            thresh_C=thresh_C,
            edge_from_mask=edge_from_mask,
            canny_low=canny_low,
            canny_high=canny_high
        )
    except Exception as e:
        print(f"Error extracting points from image: {e}")
        raise

    # Normalize
    bounds = (-4, 4, -4, 4)
    target_points = assignment.normalize_points(target_points_raw, bounds)
    
    # Save targets_only.png (04_targets_only.png for consistency)
    plt.figure(figsize=(10, 10))
    plt.scatter(target_points[:, 0], target_points[:, 1], c='r', marker='o')
    plt.title(f"Target Points Only (N={n_drones})")
    plt.axis('equal')
    plt.grid(True)
    plt.savefig(debug_dir / "04_targets_only.png")
    # Also save as targets_only.png for backward compatibility
    plt.savefig(debug_dir / "targets_only.png")
    plt.close()
    
    # Auto-tune parameters if requested
    if auto_params:
        print("Auto-tuning parameters based on target density...")
        params = analysis.auto_params_from_targets(target_points, base_params=params)
    elif params is None:
        params = {
            'm': 1.0, 'kp': 2.0, 'kd': 1.5, 'k_rep': 2.0, 'Rsafe': 0.3, 'vmax': 5.0
        }
    
    # Inject total_time for repulsion ramping
    params['total_time'] = duration
    
    print(f"Using parameters: {json.dumps(params, indent=2)}")
    
    # 2. Initial Conditions
    initial_bounds = (-5, 5, -5, 5)
    X0 = initial_conditions.initial_positions(n_drones, mode="grid", bounds=initial_bounds)
    V0 = np.zeros_like(X0)
    
    # 3. Assignment
    targets_assigned = assignment.hungarian_assign(X0, target_points)
    
    # 4. Simulation
    def target_fn(t):
        return targets_assigned
        
    state0 = np.concatenate([X0.flatten(), V0.flatten()])
    t_span = (0.0, duration)
    rhs_func = lambda t, y: dynamics.rhs(t, y, target_fn, params)
    
    print("Simulating...")
    times, states_flat = solver.solve_ivp_rk4(rhs_func, t_span, state0, dt, pbar=True)
    
    # 5. Process
    N = n_drones
    d = 2 
    n_steps = len(times)
    X_hist = states_flat[:, :N*d].reshape(n_steps, N, d)
    V_hist = states_flat[:, N*d:].reshape(n_steps, N, d)
    
    # Convergence Analysis
    diff = X_hist - targets_assigned[np.newaxis, :, :]
    dist_err = np.linalg.norm(diff, axis=2) 
    mean_error = np.mean(dist_err, axis=1) 
    max_error = np.max(dist_err, axis=1)
    final_mean_err = mean_error[-1]
    
    # Pairwise distances (min) over time
    min_pw_dist = []
    # Only calculate for a subset of steps to save time if N is large? 
    # For N=80 it's fine.
    for i in range(0, n_steps, max(1, n_steps//100)): # 100 sample points
        # O(N^2) here
        from scipy.spatial.distance import pdist
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
    
    # 6. Save
    npz_path = output_dir / "trajectories.npz"
    # Store bounds for Task 2 to use same coordinate system
    bounds_array = np.array(bounds)  # Convert tuple to array for NPZ storage
    np.savez_compressed(
        npz_path,
        times=times, X=X_hist, V=V_hist, targets=targets_assigned, params=params, 
        mean_error=mean_error, max_error=max_error, bounds=bounds_array
    )
    
    summary = {
        "n_drones": n_drones, "duration": duration, "dt": dt, 
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
        "min_pairwise_dist": min_pw_dist
    }
    with open(output_dir / "analysis.json", "w") as f:
        json.dump(analysis_data, f)
    
    # 7. Generate Preview using visualize module
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Determine bounds
    all_x = np.concatenate([X_hist[:, :, 0].flatten(), targets_assigned[:, 0]])
    all_y = np.concatenate([X_hist[:, :, 1].flatten(), targets_assigned[:, 1]])
    x_margin = (np.max(all_x) - np.min(all_x)) * 0.1
    y_margin = (np.max(all_y) - np.min(all_y)) * 0.1
    bounds = (np.min(all_x) - x_margin, np.max(all_x) + x_margin,
              np.min(all_y) - y_margin, np.max(all_y) + y_margin)
    
    # Generate title
    title_str = (f"Task 1: Static Formation (N={N})\n"
                 f"Err={final_mean_err:.3f}, MinDist={final_min_dist:.3f}, Rsafe={params['Rsafe']:.3f}")
    
    # Plot with explicit X_start and X_final
    visualize.plot_frame(ax, X=None, targets=targets_assigned, title=title_str, 
                        bounds=bounds, X_start=X_hist[0], X_final=X_hist[-1])
    
    # Add trajectory traces for a few drones
    for i in range(min(5, N)):
        ax.plot(X_hist[:, i, 0], X_hist[:, i, 1], 'k-', alpha=0.1, linewidth=1)
    
    plt.savefig(output_dir / "preview.png")
    plt.close()
    
    # 8. Generate Animation
    print("Generating animation...")
    try:
        animation_path = output_dir / "animation.mp4"
        visualize.animate_trajectories(npz_path, animation_path, fps=30, trail=0, show_targets=True)
    except Exception as e:
        print(f"Animation generation failed: {e}")
        # Continue without animation
    
    print(f"Outputs saved to {output_dir}")
    return summary
