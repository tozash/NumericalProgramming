import numpy as np
from scipy.spatial.distance import cdist

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
