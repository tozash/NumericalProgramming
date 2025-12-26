
import numpy as np

def chord_length_parameterization(x, y):
    """
    Computes cumulative chord length parameterization for a set of points (x, y).
    
    Args:
        x (array-like): x-coordinates of nodes.
        y (array-like): y-coordinates of nodes.
        
    Returns:
        t (ndarray): Normalized parameter t in [0, 1] based on cumulative distance.
    """
    x = np.array(x)
    y = np.array(y)
    
    # Euclidean distance between consecutive points
    dx = np.diff(x)
    dy = np.diff(y)
    distances = np.sqrt(dx**2 + dy**2)
    
    # Cumulative distance (starting at 0)
    # We prepend 0 to match the length of x and y
    cumulative_dist = np.concatenate(([0], np.cumsum(distances)))
    
    # Normalize to [0, 1]
    total_length = cumulative_dist[-1]
    if total_length == 0:
        # Handle single point or duplicate points edge case
        return np.zeros_like(cumulative_dist)
        
    t = cumulative_dist / total_length
    
    return t
