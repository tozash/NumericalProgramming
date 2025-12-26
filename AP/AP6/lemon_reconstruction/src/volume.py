import numpy as np

def integrate_volume(y, r):
    """
    Computes volume of revolution using the disk method.
    V = pi * integral r(y)^2 dy
    
    Uses Trapezoidal rule.
    """
    # Sort just in case
    idx = np.argsort(y)
    y_s = y[idx]
    r_s = r[idx]
    
    # Calculate cross-sectional areas
    areas = np.pi * (r_s ** 2)
    
    # Trapezoidal rule: np.trapz(y, x) -> int y dx
    volume = np.trapz(areas, y_s)
    
    # Since y might act downwards (pixels), volume might be negative if dy is negative.
    return np.abs(volume)
    
def integrate_volume_simpson(y, r):
    """
    Computes volume using Simpson's rule.
    Requires odd number of points (even number of intervals).
    """
    from scipy.integrate import simpson
    idx = np.argsort(y)
    y_s = y[idx]
    r_s = r[idx]
    areas = np.pi * (r_s ** 2)
    
    return np.abs(simpson(areas, x=y_s))
