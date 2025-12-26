import numpy as np
import cv2

def find_symmetry_axis(edge_map, search_range=None):
    """
    Finds the vertical axis of symmetry (x=x0) by minimizing mirror mismatch.
    Uses Distance Transform for mismatch score calculation.
    
    Args:
        edge_map: Binary edge map (0 or 255).
        search_range: Tuple (min_x, max_x) to search. Default: central half.
        
    Returns:
        best_x: The x-coordinate of the symmetry axis.
        scores: List of scores for visualization (optional).
    """
    h, w = edge_map.shape
    
    if search_range is None:
        search_range = (w // 4, 3 * w // 4)
        
    min_x, max_x = search_range
    best_x = min_x
    min_score = float('inf')
    
    # Distance transform: distance to nearest zero pixel (nearest background)
    # We want distance to nearest EDGE pixel.
    # So we invert edge map: edges become 0, background becomes 255.
    inverted_edges = cv2.bitwise_not(edge_map)
    dist_transform = cv2.distanceTransform(inverted_edges, cv2.DIST_L2, 5)
    
    # We will iterate and inspect the "mirrored" edges against the distance transform
    # If perfect symmetry, mirrored edges will land on 0 distance.
    
    # Only check integer x for simplicity
    for x0 in range(min_x, max_x):
        # We need to reflect the edge points across x0.
        # Efficient way:
        # 1. Get coordinates of all edge pixels
        # 2. Reflect x-coordinates: x' = 2*x0 - x
        # 3. Sum the distance transform values at (y, x')
        
        # However, purely image-based approach:
        # Create a reflected image? No, that's slow.
        # Let's try to extract edge points first.
        y_inds, x_inds = np.where(edge_map > 0)
        
        # Reflect x indices
        x_reflected = 2 * x0 - x_inds
        
        # Filter out of bounds
        valid_mask = (x_reflected >= 0) & (x_reflected < w)
        x_valid = x_reflected[valid_mask]
        y_valid = y_inds[valid_mask]
        
        if len(x_valid) == 0:
            continue
            
        # Sum distances at these valid reflected points
        # dist_transform[y, x] gives distance to nearest edge
        current_score = np.sum(dist_transform[y_valid, x_valid])
        
        # Normalize by number of points to avoid bias towards fewer points inside bounds
        current_score /= (len(x_valid) + 1e-6)
        
        if current_score < min_score:
            min_score = current_score
            best_x = x0
            
    return best_x
