
import numpy as np
from scipy.spatial.distance import cdist

def compute_metrics(fitted_x, fitted_y, original_nodes):
    """
    Computes distance metrics between the fitted curve and the original nodes.
    
    The metric is defined as the distance from each ORIGINAL reference point 
    to the NEAREST point on the densely sampled fitted curve.
    
    Args:
        fitted_x (array-like): Dense sampling of x coordinates of the fit.
        fitted_y (array-like): Dense sampling of y coordinates of the fit.
        original_nodes (list of tuples): The original (x, y) control points.
        
    Returns:
        dict: {'mean_error': float, 'max_error': float}
    """
    
    # Convert inputs to numpy arrays
    fit_points = np.column_stack((fitted_x, fitted_y))
    ref_points = np.array(original_nodes)
    
    # We want to find, for each ref_point, the min distance to any fit_point.
    # We can use cdist to get all pairwise distances.
    # Shape: (n_ref, n_fit)
    dists = cdist(ref_points, fit_points, metric='euclidean')
    
    # For each reference point (row), find the minimum distance to the curve
    min_dists = np.min(dists, axis=1)
    
    mean_err = np.mean(min_dists)
    max_err = np.max(min_dists)
    
    return {
        'mean_error': mean_err,
        'max_error': max_err
    }
