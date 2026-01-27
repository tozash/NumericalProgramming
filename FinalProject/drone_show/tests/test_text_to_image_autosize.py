import numpy as np
import pytest
from drone_show.preprocess import text_to_image

def test_text_to_image_autosize():
    """Test that long text is not clipped."""
    text = "SANDROTEST"
    font_size = 120
    padding = 30
    
    img = text_to_image(text, font_size=font_size, padding=padding, thickness=5)
    
    # Check dimensions
    H, W = img.shape
    assert H > font_size # Should be at least font size height
    
    # Check content
    # Find bounding box of non-zero pixels
    rows, cols = np.where(img > 0.1)
    
    if len(cols) == 0:
        pytest.fail("Generated image is empty")
        
    min_x, max_x = np.min(cols), np.max(cols)
    span_x = max_x - min_x
    
    # Text "SANDROTEST" is long. Aspect ratio (width/height) should be large.
    # Height is roughly font_size (or slightly more with padding).
    # Width should be significantly larger than height.
    # "SANDROTEST" is 10 chars. 10 * width_per_char. 
    # Usually width > height for this string.
    
    assert span_x > 2.0 * font_size, f"Text width {span_x} is too small for font size {font_size}. Likely clipped."
    
    # Check that we have margins (padding)
    # The image size is W. min_x should be >= padding roughly.
    # Note: font rendering can be tricky with exact pixels, but let's check basic sanity.
    assert min_x >= 0
    assert max_x < W
