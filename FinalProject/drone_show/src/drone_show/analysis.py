import numpy as np
from scipy.spatial.distance import cdist, pdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time

def median_target_spacing(T):
    """
    Computes the median nearest-neighbor distance for a set of points.
    
    Args:
        T (np.ndarray): Target points of shape (N, 2).
        
    Returns:
        float: Median NN distance.
    """
    if len(T) < 2:
        return 1.0 # Fallback
        
    # Compute full pairwise distance matrix
    dists = cdist(T, T)
    
    # Mask diagonal (dist to self is 0)
    np.fill_diagonal(dists, np.inf)
    
    # Find NN for each point
    nn_dists = np.min(dists, axis=1)
    
    return np.median(nn_dists)

def auto_params_from_targets(T, base_params=None):
    """
    Automatically tunes physics parameters based on target density.
    
    Args:
        T (np.ndarray): Target points (N, 2).
        base_params (dict): Optional base parameters to override.
        
    Returns:
        dict: Tuned parameters.
    """
    # 1. Determine spacing
    d_median = median_target_spacing(T)
    
    # 2. Defaults
    if base_params is None:
        base_params = {}
        
    params = base_params.copy()
    
    # Rsafe should be smaller than median spacing to allow packing
    # 0.6 * d ensures we don't repel too strongly from neighbors in the target formation
    if 'Rsafe' not in params:
        params['Rsafe'] = 0.6 * d_median
        
    # Standard mass
    if 'm' not in params:
        params['m'] = 1.0
    
    m = params['m']
    
    # Stiffness kp: need it strong enough to overcome repulsion at target
    # Default to 20.0 if not set
    if 'kp' not in params:
        params['kp'] = 20.0
        
    kp = params['kp']
    
    # Damping kd: Critical damping = 2 * sqrt(k*m)
    if 'kd' not in params:
        params['kd'] = 2.0 * np.sqrt(kp * m)
        
    # Repulsion k_rep
    # Should be small relative to attraction at target distances
    # Formula: 0.02 * kp * (Rsafe**3)
    # The term (1/d - 1/R) / d^2 roughly scales with 1/R^3 near boundary
    # We want F_rep(at d_median) << F_attraction(small error)
    if 'k_rep' not in params:
        params['k_rep'] = 0.05 * kp * (params['Rsafe']**3) # Slight bump from 0.02 for safety
        
    # Max speed
    if 'vmax' not in params:
        params['vmax'] = 5.0 # Increased from 2.0 to allow faster convergence
        
    return params


def error_series_against_targets(times, X, target_fn):
    """
    Computes error series against time-varying targets.
    
    Args:
        times (np.ndarray): (K,) time array.
        X (np.ndarray): (K, N, 2) drone positions over time.
        target_fn (callable): Function target_fn(t) -> (N, 2) target positions.
        
    Returns:
        tuple: (mean_error, max_error)
            - mean_error: (K,) mean error per time step.
            - max_error: (K,) max error per time step.
    """
    K = len(times)
    N = X.shape[1]
    
    mean_error = np.zeros(K)
    max_error = np.zeros(K)
    
    for i, t in enumerate(times):
        T_t = target_fn(t)
        diff = X[i] - T_t
        dist_err = np.linalg.norm(diff, axis=1)
        mean_error[i] = np.mean(dist_err)
        max_error[i] = np.max(dist_err)
    
    return mean_error, max_error


def collision_series(X, Rsafe):
    """
    Computes minimum pairwise distance series to detect collisions.
    
    Args:
        X (np.ndarray): (K, N, 2) drone positions over time.
        Rsafe (float): Safety radius.
        
    Returns:
        np.ndarray: (K,) minimum pairwise distance at each time step.
    """
    K, N, _ = X.shape
    min_dist = np.zeros(K)
    
    for i in range(K):
        if N > 1:
            dists = pdist(X[i])
            if len(dists) > 0:
                min_dist[i] = np.min(dists)
            else:
                min_dist[i] = np.inf
        else:
            min_dist[i] = np.inf
    
    return min_dist


def dt_refinement(task_runner, dt_list, out_dir, seed=42):
    """
    Runs a task with different dt values to check convergence and stability.
    
    Args:
        task_runner (callable): Function that takes dt and returns (times, X, target_fn, summary).
        dt_list (list): List of dt values to test.
        out_dir (Path): Output directory for results.
        seed (int): Random seed for reproducibility.
        
    Returns:
        dict: Results dictionary with dt, final_mean_error, runtime, etc.
    """
    from .. import utils
    utils.set_deterministic_behavior(seed)
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for dt in dt_list:
        print(f"Testing dt={dt:.4f}...")
        start_time = time.time()
        
        try:
            times, X, target_fn, summary = task_runner(dt)
            runtime = time.time() - start_time
            
            # Compute error series
            mean_error, max_error = error_series_against_targets(times, X, target_fn)
            final_mean_err = mean_error[-1]
            
            # Get Rsafe from summary if available
            Rsafe = summary.get('params', {}).get('Rsafe', 0.3)
            min_dist = collision_series(X, Rsafe)
            final_min_dist = min_dist[-1]
            
            results.append({
                'dt': dt,
                'final_mean_error': float(final_mean_err),
                'final_max_error': float(max_error[-1]),
                'final_min_dist': float(final_min_dist),
                'runtime': runtime,
                'n_steps': len(times),
                'success': True
            })
        except Exception as e:
            runtime = time.time() - start_time
            results.append({
                'dt': dt,
                'final_mean_error': np.inf,
                'final_max_error': np.inf,
                'final_min_dist': 0.0,
                'runtime': runtime,
                'n_steps': 0,
                'success': False,
                'error': str(e)
            })
    
    # Save results
    with open(out_dir / "dt_table.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Plot dt vs error
    dts = [r['dt'] for r in results if r['success']]
    errors = [r['final_mean_error'] for r in results if r['success']]
    
    if len(dts) > 0:
        plt.figure(figsize=(10, 6))
        plt.semilogx(dts, errors, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Time Step dt')
        plt.ylabel('Final Mean Error')
        plt.title('Convergence vs Time Step')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "dt_vs_error.png", dpi=150)
        plt.close()
        
        # Plot dt vs runtime
        runtimes = [r['runtime'] for r in results if r['success']]
        plt.figure(figsize=(10, 6))
        plt.loglog(dts, runtimes, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Time Step dt')
        plt.ylabel('Runtime (seconds)')
        plt.title('Runtime vs Time Step')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "dt_vs_runtime.png", dpi=150)
        plt.close()
    
    return results
