import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path

def load_image_gray(path):
    """
    Loads an image from path, converts to grayscale, and normalizes to float32 [0, 1].
    
    Args:
        path (str or Path): Path to image file.
        
    Returns:
        np.ndarray: Grayscale image of shape (H, W) with values in [0, 1].
        
    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If image cannot be loaded.
    """
    path_str = str(path)
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"Image file not found: {path_str}")
        
    # Load using OpenCV to get numpy array directly
    img = cv2.imread(path_str, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError(f"Failed to load image from {path_str}")
        
    # Normalize to [0, 1]
    return img.astype(np.float32) / 255.0

def text_to_image(text, font_size=50, padding=20, thickness=1):
    """
    Creates an image with the given text rendered.
    
    Args:
        text (str): Text to render.
        font_size (int): Font size.
        padding (int): Padding around text.
        thickness (int): Stroke width for text.
        
    Returns:
        np.ndarray: Grayscale image (H, W) in [0, 1].
    """
    # Create a dummy image to calculate text size
    dummy_img = Image.new('L', (1, 1), 0)
    draw = ImageDraw.Draw(dummy_img)
    
    try:
        # Try to load a standard font
        font_names = ["arial.ttf", "DejaVuSans.ttf", "FreeSans.ttf", "OpenSans-Regular.ttf"]
        font = None
        for fn in font_names:
            try:
                font = ImageFont.truetype(fn, font_size)
                break
            except OSError:
                continue
        
        if font is None:
            # Fallback to default if no system font found (might not scale well)
            print("Warning: Could not load system font, using default.")
            font = ImageFont.load_default()
            
    except Exception:
         font = ImageFont.load_default()
    
    # Calculate exact bounding box
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=thickness)
    
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    W = text_w + 2 * padding
    H = text_h + 2 * padding
    
    # Ensure dimensions are at least 1
    W = max(1, W)
    H = max(1, H)
    
    # Create actual image (black background)
    img = Image.new('L', (W, H), 0)
    draw = ImageDraw.Draw(img)
    
    x = padding - bbox[0]
    y = padding - bbox[1]
    
    draw.text((x, y), text, font=font, fill=255, stroke_width=thickness)
    
    # Convert to numpy
    arr = np.array(img).astype(np.float32) / 255.0
    return arr

def to_binary_mask_from_image(path):
    """
    Loads an image and creates a robust binary mask of foreground content.
    
    Args:
        path (str or Path): Path to image.
        
    Returns:
        np.ndarray: uint8 mask {0, 1} of shape (H, W).
    """
    # Load grayscale (0..255 uint8)
    path_str = str(path)
    if not os.path.exists(path_str):
        raise FileNotFoundError(f"Image not found: {path_str}")
        
    img = cv2.imread(path_str, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {path_str}")
        
    # 1. Blur lightly to reduce noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # 2. Otsu thresholding
    # This automatically finds threshold separating bimodality
    thresh_val, bin_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 3. Auto-invert check
    # Assume background is the dominant area (corner pixels usually background)
    # Check corners or mean.
    # If mean is high (> 127), likely white background -> invert.
    if np.mean(bin_img) > 127:
        bin_img = 255 - bin_img
        
    # 4. Morphological cleanup (open to remove specks, close to fill gaps)
    kernel = np.ones((3, 3), np.uint8)
    # Removing small white noise
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
    # Filling small holes in text
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
    
    # Return as 0/1 mask
    return (bin_img > 127).astype(np.uint8)

def to_binary_mask_from_text(text, font_size=50, padding=20, thickness=1):
    """
    Renders text directly to a binary mask.
    
    Returns:
        np.ndarray: uint8 mask {0, 1} of shape (H, W).
    """
    # Generate grayscale float image [0,1]
    img_float = text_to_image(text, font_size, padding, thickness)
    
    # Convert to binary mask (foreground > 0.5)
    mask = (img_float > 0.5).astype(np.uint8)
    return mask

def illumination_correct(gray, method="divide", k_frac=0.12):
    """
    Corrects uneven illumination (shadows) in grayscale image.
    
    Args:
        gray (np.ndarray): Grayscale image, float32 [0,1] or uint8 [0,255].
        method (str): "divide" or "subtract" for background removal.
        k_frac (float): Kernel size as fraction of min(H,W).
        
    Returns:
        np.ndarray: Corrected image uint8 [0,255].
    """
    # Convert to uint8 if needed
    if gray.dtype == np.float32 or gray.dtype == np.float64:
        gray_u8 = (gray * 255).astype(np.uint8)
    else:
        gray_u8 = gray.astype(np.uint8)
    
    H, W = gray_u8.shape
    
    # Compute kernel size
    k = int(k_frac * min(H, W))
    k = max(31, k)  # Minimum size
    if k % 2 == 0:
        k += 1  # Make odd
    
    # Estimate background with large blur
    bg = cv2.GaussianBlur(gray_u8, (k, k), 0)
    
    # Correct illumination
    if method == "divide":
        # Divide method: normalize by background
        corr = gray_u8.astype(np.float32) * 255.0 / (bg.astype(np.float32) + 1.0)
        corr = np.clip(corr, 0, 255).astype(np.uint8)
    elif method == "subtract":
        # Subtract method: remove background, then normalize
        corr = cv2.subtract(gray_u8, bg)
        # Normalize to [0, 255]
        corr_min, corr_max = corr.min(), corr.max()
        if corr_max > corr_min:
            corr = ((corr - corr_min) * 255.0 / (corr_max - corr_min)).astype(np.uint8)
        else:
            corr = corr.astype(np.uint8)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    corr = clahe.apply(corr)
    
    return corr

def ink_mask_from_corrected(corr_u8, mode="adaptive", block_size=35, C=10):
    """
    Extracts ink mask from illumination-corrected image.
    
    Args:
        corr_u8 (np.ndarray): Corrected image uint8 [0,255].
        mode (str): "adaptive" or "otsu" thresholding.
        block_size (int): Block size for adaptive threshold (must be odd).
        C (float): Constant subtracted from mean for adaptive threshold.
        
    Returns:
        np.ndarray: Binary mask uint8 {0, 255}.
    """
    H, W = corr_u8.shape
    
    # Thresholding
    if mode == "otsu":
        _, mask = cv2.threshold(corr_u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif mode == "adaptive":
        # Ensure block_size is odd
        if block_size % 2 == 0:
            block_size += 1
        mask = cv2.adaptiveThreshold(
            corr_u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, C
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill gaps
    
    # Remove tiny components
    img_area = H * W
    min_area = 0.0002 * img_area
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # Create filtered mask
    filtered_mask = np.zeros_like(mask)
    for label_id in range(1, num_labels):  # Skip background (0)
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_area:
            filtered_mask[labels == label_id] = 255
    
    return filtered_mask

def edges_from_mask(mask_u8, method="morph", canny_low=50, canny_high=150):
    """
    Extracts edges from a binary mask (not from raw image).
    
    Args:
        mask_u8 (np.ndarray): Binary mask uint8 {0, 255}.
        method (str): "morph" or "canny".
        canny_low (int): Low threshold for Canny (if method="canny").
        canny_high (int): High threshold for Canny (if method="canny").
        
    Returns:
        np.ndarray: Binary edge map uint8 {0, 255}.
    """
    if method == "morph":
        # Morphological gradient: edge = dilation - erosion
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(mask_u8, cv2.MORPH_GRADIENT, kernel)
        # Binarize
        edges = (edges > 0).astype(np.uint8) * 255
    elif method == "canny":
        edges = cv2.Canny(mask_u8, canny_low, canny_high)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return edges
