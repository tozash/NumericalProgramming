import numpy as np
import pytest
import cv2
from pathlib import Path
import sys
import tempfile
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from drone_show import video_tracking, video_io


def test_video_tracking_synthetic(tmp_path):
    """
    Test tracking on a synthetic video with a moving dot.
    """
    # Create a synthetic video: moving white dot on black background
    width, height = 640, 480
    fps = 30.0
    num_frames = 60
    
    # Create video path
    video_path = tmp_path / "synthetic_video.avi"
    
    # Define ground truth trajectory (moving dot)
    # Dot moves from (100, 100) to (500, 300) linearly
    start_pos = np.array([100.0, 100.0])
    end_pos = np.array([500.0, 300.0])
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    
    dot_radius = 10
    ground_truth_positions = []
    
    for frame_idx in range(num_frames):
        # Create black frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Compute dot position
        t = frame_idx / (num_frames - 1)  # 0 to 1
        pos = start_pos + t * (end_pos - start_pos)
        pos_int = pos.astype(int)
        ground_truth_positions.append(pos.copy())
        
        # Draw white dot
        cv2.circle(frame, tuple(pos_int), dot_radius, (255, 255, 255), -1)
        
        out.write(frame)
    
    out.release()
    
    # Define bbox around initial position (should contain the dot)
    bbox_size = 40
    init_x = int(start_pos[0] - bbox_size // 2)
    init_y = int(start_pos[1] - bbox_size // 2)
    init_bbox = (init_x, init_y, bbox_size, bbox_size)
    
    # Track
    times_sec, centroids_px, status_info = video_tracking.track_centroid_optical_flow(
        video_path,
        init_bbox=init_bbox,
        max_frames=None,
        stride=1,
        min_features=10  # Lower threshold for synthetic video
    )
    
    # Assertions
    assert len(times_sec) == num_frames, f"Expected {num_frames} frames, got {len(times_sec)}"
    assert len(centroids_px) == num_frames
    assert centroids_px.shape == (num_frames, 2)
    
    # Compare tracked centroids with ground truth
    ground_truth_array = np.array(ground_truth_positions)
    errors = np.linalg.norm(centroids_px - ground_truth_array, axis=1)
    mean_error = np.mean(errors)
    
    # Mean error should be small (less than 5 pixels)
    assert mean_error < 5.0, f"Mean tracking error {mean_error:.2f} pixels exceeds threshold"
    
    # Check that features were detected
    assert any(n > 0 for n in status_info['n_features_per_frame']), "No features detected"


def test_video_io_meta(tmp_path):
    """Test video metadata reading."""
    # Create a simple synthetic video
    width, height = 320, 240
    fps = 25.0
    num_frames = 50
    
    video_path = tmp_path / "test_video.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    
    for _ in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    
    # Read metadata
    read_fps, read_width, read_height, read_count = video_io.read_video_meta(video_path)
    
    assert read_width == width
    assert read_height == height
    assert read_count == num_frames
    assert abs(read_fps - fps) < 0.1  # Allow small floating point differences


def test_video_io_iter_frames(tmp_path):
    """Test frame iteration."""
    width, height = 200, 150
    fps = 10.0
    num_frames = 20
    
    video_path = tmp_path / "iter_test.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    
    # Create frames with different colors
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = (i * 10) % 256  # Vary blue channel
        out.write(frame)
    out.release()
    
    # Iterate all frames
    frames_list = list(video_io.iter_frames(video_path))
    assert len(frames_list) == num_frames
    
    # Iterate with stride
    frames_stride = list(video_io.iter_frames(video_path, stride=2))
    assert len(frames_stride) == num_frames // 2
    
    # Iterate with max_frames
    frames_max = list(video_io.iter_frames(video_path, max_frames=5))
    assert len(frames_max) == 5
