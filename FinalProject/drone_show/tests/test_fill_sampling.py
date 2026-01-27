import numpy as np
import cv2
import pytest
from drone_show import preprocess, geometry

def test_fill_sampling_covers_many_components():
    """Ensure fill sampling doesn't just clump in one letter."""
    text = "SANDROTEST"
    mask = preprocess.to_binary_mask_from_text(text, font_size=100, padding=30, thickness=4)
    
    # Identify connected components (letters)
    num_labels, labels = cv2.connectedComponents(mask)
    # label 0 is background
    
    # Sample points
    K = 200
    # Use geometry internal to sample from mask (returns float xy)
    # We need to map back to pixels to check labels.
    # sample_points_from_mask returns (x, y_flipped)
    # We need to flip y back to index into mask.
    points = geometry.sample_points_from_mask(mask, K, downsample=2)
    H, W = mask.shape
    
    # Flip Y back to image coords
    y_img = (H - 1) - points[:, 1]
    x_img = points[:, 0]
    
    # Round to nearest pixel
    y_idx = np.clip(np.round(y_img).astype(int), 0, H-1)
    x_idx = np.clip(np.round(x_img).astype(int), 0, W-1)
    
    # Get labels at these points
    hit_labels = labels[y_idx, x_idx]
    
    # Count unique labels (exclude 0)
    unique_hits = np.unique(hit_labels)
    unique_hits = unique_hits[unique_hits != 0]
    
    # "SANDROTEST" has 10 letters. "O", "R", "D", "A", "O" might have holes, but distinct components should be >= 6 roughly
    # (some letters might touch if thickness is high, but usually distinct)
    assert len(unique_hits) >= 6, f"Only hit {len(unique_hits)} components (expected >= 6 for '{text}')"

def test_fill_sampling_spread():
    """Ensure points are spread out (farthest point sampling works)."""
    # Create a mask that is a solid block
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[25:75, 25:75] = 1
    
    K = 50
    points = geometry.sample_points_from_mask(mask, K, downsample=1, method="farthest")
    
    # Normalize to [0,1] for easy threshold check
    p_norm = points / 100.0
    
    # Compute pairwise distances
    from scipy.spatial.distance import pdist
    dists = pdist(p_norm)
    
    # If points were all same, min dist would be 0.
    # If points are spread in 50x50 area (0.5x0.5), with 50 points.
    # Density ~ 50 / 0.25 = 200 pts/unit^2. Avg spacing ~ 1/sqrt(200) ~ 0.07
    # Nearest neighbor shouldn't be super tiny.
    min_dist = np.min(dists)
    
    # If random, can be very small. With farthest, should be decent.
    # Let's assert > 0.02 (very loose, but guards against duplicates/collapse)
    assert min_dist > 0.02, f"Points are too clumped (min dist {min_dist})"
    
    # Coverage check: bounding box of points should approximate mask bounds
    # Mask is 25-75 (flipped Y 24-74 approx)
    # X range
    assert np.min(points[:, 0]) >= 25
    assert np.max(points[:, 0]) <= 75 # slightly less/more due to pixel centers
    # Y range (flipped)
    # y_img 25 corresponds to y_cart 74
    # y_img 75 corresponds to y_cart 24
    assert np.min(points[:, 1]) >= 24
    assert np.max(points[:, 1]) <= 74
