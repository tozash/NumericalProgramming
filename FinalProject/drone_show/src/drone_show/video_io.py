"""
Video I/O utilities for reading video files.
"""
import cv2
import numpy as np
from pathlib import Path


def read_video_meta(path):
    """
    Reads video metadata.
    
    Args:
        path (str or Path): Path to video file.
        
    Returns:
        tuple: (fps, width, height, frame_count) as floats/ints.
        
    Raises:
        FileNotFoundError: If video file doesn't exist.
        ValueError: If video cannot be opened.
    """
    path_str = str(path)
    if not Path(path_str).exists():
        raise FileNotFoundError(f"Video file not found: {path_str}")
    
    cap = cv2.VideoCapture(path_str)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {path_str}")
    
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        return fps, width, height, frame_count
    finally:
        cap.release()


def iter_frames(path, max_frames=None, stride=1):
    """
    Iterates over video frames.
    
    Args:
        path (str or Path): Path to video file.
        max_frames (int, optional): Maximum number of frames to read.
        stride (int): Frame stride (1 = every frame, 2 = every other frame, etc.).
        
    Yields:
        tuple: (frame_index, frame_bgr) where frame_bgr is uint8 BGR image (H, W, 3).
    """
    path_str = str(path)
    cap = cv2.VideoCapture(path_str)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {path_str}")
    
    try:
        frame_idx = 0
        yielded_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only yield if stride matches
            if frame_idx % stride == 0:
                yield frame_idx, frame
                yielded_count += 1
                
                if max_frames is not None and yielded_count >= max_frames:
                    break
            
            frame_idx += 1
    finally:
        cap.release()
