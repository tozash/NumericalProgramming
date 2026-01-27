import numpy as np
import pytest
from PIL import Image, ImageDraw
import os
import cv2
from drone_show import geometry, preprocess

@pytest.fixture
def synthetic_image_path(tmp_path):
    """Creates a temporary image with a white square."""
    path = tmp_path / "test_shape.png"
    
    # Create 100x100 black image
    img = Image.new('L', (100, 100), 0)
    draw = ImageDraw.Draw(img)
    
    # Draw white square in middle: (25, 25) to (75, 75)
    # Rectangle is [x0, y0, x1, y1]
    draw.rectangle([25, 25, 75, 75], fill=255)
    
    img.save(path)
    return path

def test_extract_shape_points(synthetic_image_path):
    """Test full pipeline on synthetic square."""
    K = 100
    points = geometry.extract_shape_points_from_image(synthetic_image_path, K, smooth=False)
    
    assert points.shape == (K, 2)
    
    # Check bounds
    # The square is from 25 to 75. 
    # Canny might shift edges slightly (1-2 pixels).
    # We expect points roughly within [20, 80].
    assert np.all(points >= 20)
    assert np.all(points <= 80)
    
    # Check that we have variation in both X and Y (it's not a line)
    std_x = np.std(points[:, 0])
    std_y = np.std(points[:, 1])
    assert std_x > 10
    assert std_y > 10

def test_smooth_spline():
    """Test smoothing keeps shape and returns K points."""
    # Create a noisy circle
    t = np.linspace(0, 2*np.pi, 50)
    x = 10 * np.cos(t)
    y = 10 * np.sin(t)
    # Add noise? Not needed for basic shape check, just connectivity.
    points = np.column_stack([x, y])
    
    K = 200
    smoothed = geometry.smooth_contour_spline(points, K)
    
    assert smoothed.shape == (K, 2)
    assert not np.any(np.isnan(smoothed))
    
    # Check it's roughly circular (bounds approx -10 to 10)
    assert np.all(smoothed >= -11)
    assert np.all(smoothed <= 11)

def test_text_to_image():
    """Check text generation works."""
    img = preprocess.text_to_image("A", font_size=20)
    assert isinstance(img, np.ndarray)
    assert img.ndim == 2
    assert img.max() <= 1.0
    assert img.min() >= 0.0
    
    # Should have some white pixels
    assert np.sum(img > 0.5) > 0

def test_sample_polyline_uniform():
    """Check uniform sampling of a line."""
    # Line from (0,0) to (10,0)
    points = np.array([[0.0, 0.0], [10.0, 0.0]])
    K = 11
    sampled = geometry.sample_polyline_uniform(points, K)
    
    assert sampled.shape == (K, 2)
    
    # Should be 0, 1, 2, ..., 10
    expected_x = np.linspace(0, 10, K)
    assert np.allclose(sampled[:, 0], expected_x)
    assert np.allclose(sampled[:, 1], 0)
