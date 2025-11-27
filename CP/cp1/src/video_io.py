# cp1/src/video_io.py

import cv2
import numpy as np
from typing import Generator, Tuple

def read_video_frames(video_path: str, resize_dim: Tuple[int, int] = None) -> Generator[np.ndarray, None, None]:
    """
    Generator that yields frames from a video file.
    
    Args:
        video_path: Path to the video file.
        resize_dim: Tuple (width, height) to resize frames to.
        
    Yields:
        np.ndarray: The next frame (BGR format).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if resize_dim:
            frame = cv2.resize(frame, resize_dim)
            
        yield frame

    cap.release()

def get_video_info(video_path: str) -> dict:
    """Returns metadata about the video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}
    
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }
    cap.release()
    return info

