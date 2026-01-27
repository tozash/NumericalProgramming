import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def normalize_points(points, bounds=(-1, 1, -1, 1)):
    """
    Centers and scales points to fit within bounds while preserving aspect ratio.
    
    Args:
        points (np.ndarray): Shape (N, 2).
        bounds (tuple): (xmin, xmax, ymin, ymax).
        
    Returns:
        np.ndarray: Normalized points (N, 2).
    """
    if len(points) == 0:
        return points
        
    # Current bounding box
    p_min = np.min(points, axis=0)
    p_max = np.max(points, axis=0)
    p_center = (p_min + p_max) / 2.0
    p_size = p_max - p_min
    
    # Target bounds
    xmin, xmax, ymin, ymax = bounds
    t_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
    t_size = np.array([xmax - xmin, ymax - ymin])
    
    # Determine scaling factor
    # Avoid division by zero if points are all same (size=0)
    if np.all(p_size < 1e-9):
        # All points are the same; just center them
        return points - p_center + t_center
        
    # Scale to fit the tightest dimension
    # safe division
    scale_x = t_size[0] / p_size[0] if p_size[0] > 1e-9 else float('inf')
    scale_y = t_size[1] / p_size[1] if p_size[1] > 1e-9 else float('inf')
    
    scale = min(scale_x, scale_y)
    
    # If scale is inf (shouldn't happen with check above), set to 1
    if scale == float('inf'):
        scale = 1.0
        
    # Center points -> Scale -> Move to target center
    points_centered = points - p_center
    points_scaled = points_centered * scale
    points_final = points_scaled + t_center
    
    return points_final

def hungarian_assign(X0, target_points):
    """
    Assigns current drone positions X0 to target_points to minimize total squared distance.
    
    Args:
        X0 (np.ndarray): Current positions (N, 2).
        target_points (np.ndarray): Target positions (N, 2).
        
    Returns:
        np.ndarray: Reordered target_points (N, 2) such that 
                    target_points_reordered[i] is the target for drone i.
    """
    N = X0.shape[0]
    if target_points.shape[0] != N:
        raise ValueError(f"Mismatch in number of points: X0 has {N}, targets has {target_points.shape[0]}")
        
    # Compute cost matrix (squared Euclidean distance)
    # cdist returns Euclidean distance, so we square it
    # But linear_sum_assignment works with any monotonic cost, so just dist is fine too.
    # Squared distance penalizes outliers more, which is often good.
    cost_matrix = cdist(X0, target_points, metric='sqeuclidean')
    
    # Solve assignment problem
    # row_ind maps to X0 indices (0..N-1)
    # col_ind maps to target_points indices
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # We want to return targets sorted by drone index.
    # row_ind is usually [0, 1, ..., N-1], but let's be safe.
    # We need to construct the result array.
    # result[row_ind[i]] = target_points[col_ind[i]]
    
    sorted_targets = np.zeros_like(target_points)
    sorted_targets[row_ind] = target_points[col_ind]
    
    return sorted_targets
