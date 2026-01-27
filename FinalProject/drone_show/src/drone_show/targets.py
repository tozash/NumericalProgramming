"""
Target formation utilities for time-varying formations.
"""
import numpy as np
from scipy.interpolate import interp1d


def make_centroid_interpolator(times_sec, centroids_sim):
    """
    Creates a linear interpolator for centroid positions with clamping outside range.
    
    Args:
        times_sec (np.ndarray): (K,) timestamps in seconds.
        centroids_sim (np.ndarray): (K, 2) centroid positions in simulation coordinates.
        
    Returns:
        callable: Function c(t) -> (2,) that returns centroid position at time t.
                 Clamps to first/last value outside the time range.
    """
    if len(times_sec) == 0:
        raise ValueError("times_sec must not be empty")
    if len(times_sec) == 1:
        # Single point: return constant
        c0 = centroids_sim[0]
        return lambda t: c0.copy()
    
    # Sort by time (should already be sorted, but ensure it)
    sort_idx = np.argsort(times_sec)
    times_sorted = times_sec[sort_idx]
    centroids_sorted = centroids_sim[sort_idx]
    
    # Create interpolators for x and y separately
    interp_x = interp1d(times_sorted, centroids_sorted[:, 0], 
                        kind='linear', bounds_error=False, 
                        fill_value=(centroids_sorted[0, 0], centroids_sorted[-1, 0]))
    interp_y = interp1d(times_sorted, centroids_sorted[:, 1], 
                        kind='linear', bounds_error=False, 
                        fill_value=(centroids_sorted[0, 1], centroids_sorted[-1, 1]))
    
    def c_of_t(t):
        """Returns centroid position at time t."""
        return np.array([interp_x(t), interp_y(t)])
    
    return c_of_t


def make_rigid_translation_targets(P_ref, c_of_t):
    """
    Creates a target function that performs rigid translation of a reference formation.
    
    The formation is translated so its centroid follows c(t), preserving all pairwise distances.
    
    Args:
        P_ref (np.ndarray): (N, 2) reference formation points.
        c_of_t (callable): Function c(t) -> (2,) returning centroid position at time t.
        
    Returns:
        tuple: (target_fn, sample_T_series)
            - target_fn: Function target_fn(t) -> (N, 2) returning target positions at time t.
            - sample_T_series: Function sample_T_series(times) -> (K, N, 2) returning targets at multiple times.
    """
    # Compute reference centroid
    c_ref = np.mean(P_ref, axis=0)
    
    # Compute relative positions (formation shape relative to centroid)
    P_rel = P_ref - c_ref
    
    def target_fn(t):
        """
        Returns target positions at time t.
        
        Args:
            t (float): Time in seconds.
            
        Returns:
            np.ndarray: (N, 2) target positions.
        """
        c_t = c_of_t(t)
        T_t = c_t + P_rel
        return T_t
    
    def sample_T_series(times):
        """
        Samples target positions at multiple times.
        
        Args:
            times (np.ndarray): (K,) array of times.
            
        Returns:
            np.ndarray: (K, N, 2) target positions.
        """
        K = len(times)
        N = len(P_ref)
        T_series = np.zeros((K, N, 2))
        for i, t in enumerate(times):
            T_series[i] = target_fn(t)
        return T_series
    
    return target_fn, sample_T_series
