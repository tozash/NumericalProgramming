import numpy as np
import pytest
import cv2
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from drone_show import preprocess


def test_shadow_robust_mask():
    """
    Test that shadow correction produces reasonable masks even with synthetic shadows.
    """
    # Create a synthetic handwriting-like image
    # White background with black text
    W, H = 400, 200
    img = Image.new('L', (W, H), 255)  # White background
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 40)
    except:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 40)
        except:
            font = ImageFont.load_default()
    
    # Draw text "SANDROTEST"
    text = "SANDROTEST"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (W - text_w) // 2
    y = (H - text_h) // 2
    draw.text((x, y), text, font=font, fill=0)  # Black text
    
    # Convert to numpy array
    img_arr = np.array(img).astype(np.float32) / 255.0
    
    # Add a shadow gradient (simulate half-page shadow)
    # Create a left-to-right gradient: multiply by gradient
    gradient = np.linspace(0.4, 1.0, W).reshape(1, -1)  # Dark on left, bright on right
    gradient = np.tile(gradient, (H, 1))
    img_shadowed = img_arr * gradient
    
    # Convert to uint8 for processing
    img_shadowed_u8 = (img_shadowed * 255).astype(np.uint8)
    
    # Apply illumination correction
    corr_u8 = preprocess.illumination_correct(img_shadowed, method="divide", k_frac=0.12)
    
    # Extract ink mask
    mask_u8 = preprocess.ink_mask_from_corrected(corr_u8, mode="adaptive", block_size=35, C=10)
    
    # Assertions
    # 1. Mask foreground pixel ratio is reasonable
    mask_ratio = np.sum(mask_u8 > 127) / (H * W)
    assert 0.005 < mask_ratio < 0.20, f"Mask ratio {mask_ratio} not in expected range"
    
    # 2. Connected components: should have multiple components (letters/strokes)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    # num_labels includes background (0), so we want at least 3 total (background + 2+ components)
    assert num_labels >= 3, f"Expected at least 3 connected components, got {num_labels}"
    
    # 3. Test edges_from_mask: should NOT produce a giant edge along mid-page
    edges = preprocess.edges_from_mask(mask_u8, method="morph")
    
    # Find largest connected edge component
    num_edge_labels, edge_labels, edge_stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
    if num_edge_labels > 1:  # Has at least one edge component
        edge_areas = edge_stats[1:, cv2.CC_STAT_AREA]  # Skip background
        total_edge_pixels = np.sum(edges > 127)
        
        if total_edge_pixels > 0:
            largest_edge_area = np.max(edge_areas)
            largest_fraction = largest_edge_area / total_edge_pixels
            # Largest component should be less than 15% of total edge pixels
            # (if shadow was detected as edge, it would be a huge component)
            assert largest_fraction < 0.15, f"Largest edge component {largest_fraction:.3f} too large (shadow detected?)"
    
    # 4. Test with Canny edges too
    edges_canny = preprocess.edges_from_mask(mask_u8, method="canny", canny_low=50, canny_high=150)
    assert edges_canny.shape == (H, W)
    assert edges_canny.dtype == np.uint8


def test_illumination_correct_methods():
    """Test both divide and subtract methods."""
    # Create a simple test image with shadow
    H, W = 100, 100
    img = np.ones((H, W), dtype=np.float32) * 0.5  # Gray background
    
    # Add a gradient shadow
    gradient = np.linspace(0.3, 0.7, W).reshape(1, -1)
    gradient = np.tile(gradient, (H, 1))
    img_shadowed = img * gradient
    
    # Test divide method
    corr_divide = preprocess.illumination_correct(img_shadowed, method="divide", k_frac=0.12)
    assert corr_divide.dtype == np.uint8
    assert corr_divide.shape == (H, W)
    assert np.all(corr_divide >= 0) and np.all(corr_divide <= 255)
    
    # Test subtract method
    corr_subtract = preprocess.illumination_correct(img_shadowed, method="subtract", k_frac=0.12)
    assert corr_subtract.dtype == np.uint8
    assert corr_subtract.shape == (H, W)
    assert np.all(corr_subtract >= 0) and np.all(corr_subtract <= 255)


def test_ink_mask_modes():
    """Test both adaptive and otsu thresholding modes."""
    # Create a test image
    H, W = 100, 100
    img = np.ones((H, W), dtype=np.uint8) * 200  # Light background
    
    # Add some dark text-like regions
    img[40:60, 20:80] = 50  # Dark rectangle
    
    # Test adaptive
    mask_adaptive = preprocess.ink_mask_from_corrected(img, mode="adaptive", block_size=35, C=10)
    assert mask_adaptive.dtype == np.uint8
    assert mask_adaptive.shape == (H, W)
    assert np.all((mask_adaptive == 0) | (mask_adaptive == 255))
    
    # Test otsu
    mask_otsu = preprocess.ink_mask_from_corrected(img, mode="otsu")
    assert mask_otsu.dtype == np.uint8
    assert mask_otsu.shape == (H, W)
    assert np.all((mask_otsu == 0) | (mask_otsu == 255))
