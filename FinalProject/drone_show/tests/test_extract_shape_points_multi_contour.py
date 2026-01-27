import numpy as np
import pytest
from PIL import Image
from drone_show import geometry, preprocess

def test_extract_shape_points_multi_contour(tmp_path):
    """Test extracting points from multiple contours (long text)."""
    text = "SANDROTEST"
    font_size = 50
    img_arr = preprocess.text_to_image(text, font_size=font_size, padding=20, thickness=3)
    
    # Save temp image
    path = tmp_path / "long_text.png"
    Image.fromarray((img_arr * 255).astype(np.uint8)).save(path)
    
    K = 200
    points = geometry.extract_shape_points_from_image(path, K=K, smooth=True)
    
    assert points.shape == (K, 2)
    assert not np.any(np.isnan(points))
    
    # Check X span vs Y span
    # Should be wide
    min_xy = np.min(points, axis=0)
    max_xy = np.max(points, axis=0)
    span = max_xy - min_xy
    
    # "SANDROTEST" is much wider than tall
    assert span[0] > 2.0 * span[1], f"Shape is not wide enough ({span}), implying partial extraction."
    
    # Check density/coverage
    # If we only extracted "S", the points would be clumped.
    # We expect points spread out in X.
    # Let's check std dev or histogram buckets if needed, but span check is good first proxy.
    
    # Check that we have multiple distinct components?
    # Hard to check on points cloud easily without clustering.
    # But ensuring span is wide guards against "clipping to first letter".
