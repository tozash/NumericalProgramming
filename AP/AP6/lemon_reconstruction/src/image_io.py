import cv2
import numpy as np

def load_image(path):
    """Loads an image from the specified path."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {path}")
    return img

def preprocess_image(img, blur_ksize=(5, 5)):
    """
    Converts image to grayscale and applies Gaussian blur.
    
    Args:
        img: Input image (BGR).
        blur_ksize: Kernel size for Gaussian blur.
        
    Returns:
        Preprocessed grayscale image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, blur_ksize, 0)
    return blurred
