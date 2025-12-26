import numpy as np
import cv2

def compute_gradients(image):
    """
    Computes image gradients Ix and Iy using central finite differences.
    
    Args:
        image: Grayscale input image (2D numpy array).
        
    Returns:
        Ix, Iy: Gradient images in x and y directions.
    """
    # Central difference kernels
    kernel_x = np.array([[-0.5, 0, 0.5]])
    kernel_y = np.array([[-0.5], [0], [0.5]])
    
    # Using cv2.filter2D which is equivalent to convolution/correlation
    Ix = cv2.filter2D(image.astype(float), -1, kernel_x)
    Iy = cv2.filter2D(image.astype(float), -1, kernel_y)
    
    return Ix, Iy

def compute_magnitude(Ix, Iy):
    """Computes gradient magnitude."""
    return np.sqrt(Ix**2 + Iy**2)

def get_edge_map(magnitude, threshold=None):
    """
    Thresholds the gradient magnitude to create a binary edge map.
    
    Args:
        magnitude: Gradient magnitude map.
        threshold: Float threshold. If None, uses mean + std.
        
    Returns:
        Binary edge map (0 or 255).
    """
    if threshold is None:
        # Auto-thresholding heuristic
        threshold = np.mean(magnitude) + np.std(magnitude)
        
    _, edge_map = cv2.threshold(magnitude, threshold, 255, cv2.THRESH_BINARY)
    return edge_map.astype(np.uint8)
