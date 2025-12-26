import numpy as np

def extract_profile(edge_map, axis_x):
    """
    Extracts the radial profile r(y) from the edge map and symmetry axis.
    
    Args:
        edge_map: Binary edge map.
        axis_x: x-coordinate of symmetry axis.
        
    Returns:
        y_coords, radii: Arrays of y coordinates and corresponding radii.
    """
    h, w = edge_map.shape
    y_coords = []
    radii = []
    
    # Scan each row
    for y in range(h):
        # FIND edges in this row
        row_edges = np.where(edge_map[y, :] > 0)[0]
        
        if len(row_edges) == 0:
            continue
            
        # We assume the lemon is the OBJECT.
        # We want the OUTERMOST edge relative to the axis.
        # Ideally, there's an edge on the left and right.
        # Or we act on the whole set.
        
        # Calculate distances from axis
        dists = np.abs(row_edges - axis_x)
        
        # Take the maximum distance found in this row as the radius
        # This assumes the lemon is the widest object in the row.
        r = np.max(dists)
        
        # Simple noise filter: ignore extremely small radii (near detection noise)
        if r < 2:
            continue
            
        y_coords.append(y)
        radii.append(r)
        
    y_coords = np.array(y_coords)
    radii = np.array(radii)
    
    # Sort by Y just in case
    sort_idx = np.argsort(y_coords)
    y_coords = y_coords[sort_idx]
    radii = radii[sort_idx]
    
    # Orientation: define y=0 at the bottom of the lemon
    # Image y increases downwards. Let's invert it so y=0 is bottom.
    # Actually, easy way: just use image y, but user might prefer standard math coords.
    # Let's keep image coordinates for simplicity in matching, 
    # but maybe flip for plotting if needed.
    # For now, return raw image y-coordinates.
    
    return y_coords, radii
