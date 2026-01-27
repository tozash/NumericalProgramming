import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from drone_show import video_tracking


def test_centroids_mapping_basic():
    """Test basic coordinate mapping."""
    width, height = 640, 480
    bounds = (-1, 1, -1, 1)
    
    # Test corner cases
    # Top-left corner (0, 0) -> should map to (xmin, ymax)
    centroids_px = np.array([[0, 0]])
    centroids_sim = video_tracking.centroids_px_to_sim(centroids_px, width, height, bounds)
    assert np.isclose(centroids_sim[0, 0], -1.0)  # xmin
    assert np.isclose(centroids_sim[0, 1], 1.0)   # ymax (flipped)
    
    # Bottom-right corner (width-1, height-1) -> should map to (xmax, ymin)
    centroids_px = np.array([[width - 1, height - 1]])
    centroids_sim = video_tracking.centroids_px_to_sim(centroids_px, width, height, bounds)
    assert np.isclose(centroids_sim[0, 0], 1.0)   # xmax
    assert np.isclose(centroids_sim[0, 1], -1.0)  # ymin (flipped)
    
    # Center -> should map to (0, 0)
    # Use (width-1)/2 and (height-1)/2 for exact center
    centroids_px = np.array([[(width - 1) / 2, (height - 1) / 2]])
    centroids_sim = video_tracking.centroids_px_to_sim(centroids_px, width, height, bounds)
    assert np.isclose(centroids_sim[0, 0], 0.0, atol=1e-6)
    assert np.isclose(centroids_sim[0, 1], 0.0, atol=1e-6)


def test_centroids_mapping_y_flip():
    """
    Test that Y-axis is correctly flipped (video Y increases downward,
    simulation Y increases upward).
    """
    width, height = 100, 100
    bounds = (-1, 1, -1, 1)
    
    # Top of image (y=0) should map to ymax
    centroids_px = np.array([[50, 0]])
    centroids_sim = video_tracking.centroids_px_to_sim(centroids_px, width, height, bounds)
    assert centroids_sim[0, 1] > 0, "Top of image should map to positive Y (up)"
    
    # Bottom of image (y=height-1) should map to ymin
    centroids_px = np.array([[50, height - 1]])
    centroids_sim = video_tracking.centroids_px_to_sim(centroids_px, width, height, bounds)
    assert centroids_sim[0, 1] < 0, "Bottom of image should map to negative Y (down)"


def test_centroids_mapping_custom_bounds():
    """Test mapping with custom bounds."""
    width, height = 200, 150
    bounds = (-4, 4, -3, 3)  # Custom bounds
    
    # Top-left
    centroids_px = np.array([[0, 0]])
    centroids_sim = video_tracking.centroids_px_to_sim(centroids_px, width, height, bounds)
    assert np.isclose(centroids_sim[0, 0], -4.0)  # xmin
    assert np.isclose(centroids_sim[0, 1], 3.0)   # ymax
    
    # Bottom-right
    centroids_px = np.array([[width - 1, height - 1]])
    centroids_sim = video_tracking.centroids_px_to_sim(centroids_px, width, height, bounds)
    assert np.isclose(centroids_sim[0, 0], 4.0)    # xmax
    assert np.isclose(centroids_sim[0, 1], -3.0)   # ymin


def test_centroids_mapping_multiple_points():
    """Test mapping multiple points at once."""
    width, height = 100, 100
    bounds = (-1, 1, -1, 1)
    
    # Multiple points
    centroids_px = np.array([
        [0, 0],           # Top-left
        [width-1, 0],     # Top-right
        [0, height-1],    # Bottom-left
        [width-1, height-1],  # Bottom-right
        [(width-1)/2, (height-1)/2]   # Center
    ])
    
    centroids_sim = video_tracking.centroids_px_to_sim(centroids_px, width, height, bounds)
    
    assert centroids_sim.shape == (5, 2)
    
    # Check top-left
    assert np.isclose(centroids_sim[0, 0], -1.0)
    assert np.isclose(centroids_sim[0, 1], 1.0)
    
    # Check top-right
    assert np.isclose(centroids_sim[1, 0], 1.0)
    assert np.isclose(centroids_sim[1, 1], 1.0)
    
    # Check bottom-left
    assert np.isclose(centroids_sim[2, 0], -1.0)
    assert np.isclose(centroids_sim[2, 1], -1.0)
    
    # Check bottom-right
    assert np.isclose(centroids_sim[3, 0], 1.0)
    assert np.isclose(centroids_sim[3, 1], -1.0)
    
    # Check center
    assert np.isclose(centroids_sim[4, 0], 0.0, atol=1e-6)
    assert np.isclose(centroids_sim[4, 1], 0.0, atol=1e-6)
