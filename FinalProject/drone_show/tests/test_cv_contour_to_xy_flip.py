import numpy as np
from drone_show import geometry

def test_cv_contour_to_xy_flip():
    """Test Y-flipping coordinate conversion."""
    H = 100
    
    # Case 1: Simple points
    # (0,0) in image (top-left) -> (0, 99) in Cartesian
    contour = np.array([[[0, 0]]]) # Shape (1, 1, 2)
    xy = geometry.cv_contour_to_xy(contour, H)
    
    assert np.allclose(xy[0], [0, 99])
    
    # (10, 99) in image (bottom-left area) -> (10, 0) in Cartesian
    contour = np.array([[[10, 99]]])
    xy = geometry.cv_contour_to_xy(contour, H)
    
    assert np.allclose(xy[0], [10, 0])
    
    # Case 2: Array shape (M, 2) input
    contour_flat = np.array([[0, 0], [10, 99]])
    xy_flat = geometry.cv_contour_to_xy(contour_flat, H)
    
    assert np.allclose(xy_flat[0], [0, 99])
    assert np.allclose(xy_flat[1], [10, 0])
