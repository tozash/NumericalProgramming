import numpy as np
import pytest
from pathlib import Path
import sys
import cv2
from PIL import Image, ImageDraw, ImageFont

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from drone_show import geometry


def test_handwriting_pipeline_debug_outputs(tmp_path):
    """
    Test that handwriting extraction with shadow correction produces debug outputs.
    """
    # Create a temporary handwriting-like image
    W, H = 300, 150
    img = Image.new('L', (W, H), 255)  # White background
    draw = ImageDraw.Draw(img)
    
    # Draw some text/strokes
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 30)
        except:
            font = ImageFont.load_default()
    
    text = "TEST"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (W - text_w) // 2
    y = (H - text_h) // 2
    draw.text((x, y), text, font=font, fill=0)  # Black text
    
    # Save to temp file
    img_path = tmp_path / "test_handwriting.png"
    img.save(img_path)
    
    # Collect debug artifacts
    debug_artifacts = {}
    
    def debug_callback(artifacts):
        debug_artifacts.update(artifacts)
    
    # Extract points with shadow correction enabled
    K = 50
    points = geometry.extract_shape_points_from_image(
        str(img_path),
        K=K,
        sampling="fill",
        shadow_correct=True,
        shadow_k_frac=0.12,
        shadow_method="divide",
        thresh_mode="adaptive",
        thresh_block_size=35,
        thresh_C=10,
        debug_callback=debug_callback
    )
    
    # Assertions
    assert points.shape == (K, 2), f"Expected points shape ({K}, 2), got {points.shape}"
    
    # Check that debug artifacts were collected
    assert 'gray' in debug_artifacts, "Debug callback should provide 'gray' image"
    assert 'corr' in debug_artifacts, "Debug callback should provide 'corr' image"
    assert 'mask' in debug_artifacts, "Debug callback should provide 'mask' image"
    
    # Verify artifact shapes and types
    gray = debug_artifacts['gray']
    assert gray.dtype == np.uint8, f"Gray should be uint8, got {gray.dtype}"
    assert gray.shape == (H, W), f"Gray shape should be ({H}, {W}), got {gray.shape}"
    
    corr = debug_artifacts['corr']
    assert corr.dtype == np.uint8, f"Corr should be uint8, got {corr.dtype}"
    assert corr.shape == (H, W), f"Corr shape should be ({H}, {W}), got {corr.shape}"
    
    mask = debug_artifacts['mask']
    # Mask can be uint8 {0,255} or float {0,1}
    assert mask.shape == (H, W), f"Mask shape should be ({H}, {W}), got {mask.shape}"


def test_handwriting_pipeline_edge_mode_debug(tmp_path):
    """
    Test edge mode also produces debug outputs including edges.
    """
    # Create a temporary image
    W, H = 200, 100
    img = Image.new('L', (W, H), 255)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 25)
    except:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 25)
        except:
            font = ImageFont.load_default()
    
    text = "HI"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (W - text_w) // 2
    y = (H - text_h) // 2
    draw.text((x, y), text, font=font, fill=0)
    
    img_path = tmp_path / "test_edge.png"
    img.save(img_path)
    
    debug_artifacts = {}
    
    def debug_callback(artifacts):
        debug_artifacts.update(artifacts)
    
    # Extract with edge mode and shadow correction
    K = 30
    points = geometry.extract_shape_points_from_image(
        str(img_path),
        K=K,
        sampling="edge",
        shadow_correct=True,
        edge_from_mask="morph",
        debug_callback=debug_callback
    )
    
    assert points.shape == (K, 2)
    
    # Should have edges in debug artifacts
    assert 'edges' in debug_artifacts, "Edge mode should provide 'edges' in debug"
    edges = debug_artifacts['edges']
    assert edges.dtype == np.uint8
    assert edges.shape == (H, W)
