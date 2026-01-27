"""
Video tracking using optical flow.
"""
import cv2
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
from . import video_io


def select_roi_first_frame(path):
    """
    Interactive ROI selection on the first frame of a video.
    
    Args:
        path (str or Path): Path to video file.
        
    Returns:
        tuple: (x, y, w, h) as integers, bounding box coordinates.
        
    Raises:
        ValueError: If video cannot be opened or no ROI selected.
    """
    path_str = str(path)
    cap = cv2.VideoCapture(path_str)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {path_str}")
    
    try:
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Failed to read first frame")
        
        # Select ROI
        bbox = cv2.selectROI("Select ROI (press SPACE or ENTER to confirm, ESC to cancel)", frame, False)
        cv2.destroyAllWindows()
        
        if bbox[2] == 0 or bbox[3] == 0:
            raise ValueError("No ROI selected or invalid ROI")
        
        return tuple(int(x) for x in bbox)  # (x, y, w, h)
    finally:
        cap.release()


def track_centroid_optical_flow(path, init_bbox, max_frames=None, stride=1, min_features=30):
    """
    Tracks object centroid using Lucas-Kanade optical flow.
    
    Args:
        path (str or Path): Path to video file.
        init_bbox (tuple): Initial bounding box (x, y, w, h).
        max_frames (int, optional): Maximum number of frames to process.
        stride (int): Frame stride.
        min_features (int): Minimum number of features to maintain. Re-seed if below.
        
    Returns:
        tuple: (times_sec, centroids_px, status_info)
            - times_sec: (K,) array of timestamps in seconds
            - centroids_px: (K, 2) array of centroid positions in pixels
            - status_info: dict with 'n_features_per_frame' (list) and 'reseed_frames' (list)
    """
    path_str = str(path)
    fps, width, height, _ = video_io.read_video_meta(path_str)
    dt_frame = 1.0 / fps * stride
    
    cap = cv2.VideoCapture(path_str)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {path_str}")
    
    # LK parameters
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    # Feature detection parameters
    feature_params = dict(
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=10,
        blockSize=7
    )
    
    times_sec = []
    centroids_px = []
    n_features_per_frame = []
    reseed_frames = []
    
    x, y, w, h = init_bbox
    current_bbox = [x, y, w, h]
    
    prev_frame = None
    prev_points = None
    
    frame_idx = 0
    processed_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process if stride matches
            if frame_idx % stride != 0:
                frame_idx += 1
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is None:
                # First frame: detect features in bbox
                x, y, w, h = current_bbox
                x, y, w, h = int(x), int(y), int(w), int(h)
                roi_mask = np.zeros_like(gray)
                roi_mask[y:y+h, x:x+w] = 255
                
                points = cv2.goodFeaturesToTrack(gray, mask=roi_mask, **feature_params)
                
                if points is not None and len(points) > 0:
                    prev_points = points.reshape(-1, 2)
                    # Initial centroid is center of bbox
                    centroid = np.array([x + w/2, y + h/2])
                else:
                    # Fallback: use bbox center
                    prev_points = np.array([[x + w/2, y + h/2]])
                    centroid = np.array([x + w/2, y + h/2])
                
                times_sec.append(frame_idx * dt_frame)
                centroids_px.append(centroid.copy())
                n_features_per_frame.append(len(prev_points))
                
                prev_frame = gray.copy()
                frame_idx += 1
                processed_count += 1
                
                if max_frames is not None and processed_count >= max_frames:
                    break
                continue
            
            # Track points using optical flow
            next_points, status, err = cv2.calcOpticalFlowPyrLK(
                prev_frame, gray, prev_points, None, **lk_params
            )
            
            # Filter valid points
            valid_mask = status.ravel() == 1
            valid_prev = prev_points[valid_mask]
            valid_next = next_points[valid_mask]
            
            if len(valid_next) < min_features:
                # Re-seed features in current bbox
                x, y, w, h = current_bbox
                x, y, w, h = int(x), int(y), int(w), int(h)
                roi_mask = np.zeros_like(gray)
                roi_mask[y:y+h, x:x+w] = 255
                
                new_points = cv2.goodFeaturesToTrack(gray, mask=roi_mask, **feature_params)
                
                if new_points is not None and len(new_points) > 0:
                    prev_points = new_points.reshape(-1, 2)
                    reseed_frames.append(frame_idx)
                else:
                    # Keep using previous points even if few
                    prev_points = valid_next if len(valid_next) > 0 else prev_points
            else:
                prev_points = valid_next
            
            # Compute median displacement
            if len(valid_prev) > 0 and len(valid_next) > 0:
                displacements = valid_next - valid_prev
                median_disp = np.median(displacements, axis=0)
                
                # Update centroid and bbox
                prev_centroid = centroids_px[-1]
                new_centroid = prev_centroid + median_disp
                
                # Update bbox by same displacement
                current_bbox[0] += median_disp[0]
                current_bbox[1] += median_disp[1]
            else:
                # No valid points, keep previous centroid
                if len(centroids_px) > 0:
                    new_centroid = centroids_px[-1]
                else:
                    # Fallback to bbox center
                    x, y, w, h = current_bbox
                    new_centroid = np.array([x + w/2, y + h/2])
            
            times_sec.append(frame_idx * dt_frame)
            centroids_px.append(new_centroid.copy())
            n_features_per_frame.append(len(prev_points))
            
            prev_frame = gray.copy()
            frame_idx += 1
            processed_count += 1
            
            if max_frames is not None and processed_count >= max_frames:
                break
    
    finally:
        cap.release()
    
    times_sec = np.array(times_sec)
    centroids_px = np.array(centroids_px)
    
    status_info = {
        'n_features_per_frame': n_features_per_frame,
        'reseed_frames': reseed_frames
    }
    
    return times_sec, centroids_px, status_info


def centroids_px_to_sim(centroids_px, width, height, bounds=(-1, 1, -1, 1)):
    """
    Maps pixel coordinates to simulation coordinates.
    
    Args:
        centroids_px (np.ndarray): (K, 2) pixel coordinates.
        width (int): Video width in pixels.
        height (int): Video height in pixels.
        bounds (tuple): (xmin, xmax, ymin, ymax) simulation bounds.
        
    Returns:
        np.ndarray: (K, 2) simulation coordinates.
    """
    xmin, xmax, ymin, ymax = bounds
    
    # Map x: [0, width-1] -> [xmin, xmax]
    # Use (width-1) as the maximum to map exactly to xmax
    x_sim = (centroids_px[:, 0] / (width - 1)) * (xmax - xmin) + xmin
    
    # Map y: [0, height-1] -> [ymax, ymin] (flip so up is positive)
    # Use (height-1) as the maximum to map exactly to ymin
    y_sim = (1.0 - centroids_px[:, 1] / (height - 1)) * (ymax - ymin) + ymin
    
    return np.column_stack([x_sim, y_sim])


def save_debug_outputs(path, init_bbox, times_sec, centroids_px, centroids_sim, 
                       status_info, output_dir):
    """
    Saves debug outputs for video tracking.
    
    Args:
        path (str or Path): Path to video file.
        init_bbox (tuple): Initial bounding box (x, y, w, h).
        times_sec (np.ndarray): (K,) timestamps in seconds.
        centroids_px (np.ndarray): (K, 2) pixel coordinates.
        centroids_sim (np.ndarray): (K, 2) simulation coordinates.
        status_info (dict): Status information from tracking.
        output_dir (str or Path): Output directory for debug files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read first frame
    fps, width, height, _ = video_io.read_video_meta(path)
    cap = cv2.VideoCapture(str(path))
    ret, first_frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError("Failed to read first frame for debug output")
    
    # 1. first_frame.png with bbox drawn
    first_frame_bbox = first_frame.copy()
    x, y, w, h = init_bbox
    cv2.rectangle(first_frame_bbox, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(first_frame_bbox, "Initial ROI", (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imwrite(str(output_dir / "first_frame.png"), first_frame_bbox)
    
    # 2. tracked_path.png (centroid path over first frame or blank canvas)
    path_img = first_frame.copy()
    if len(centroids_px) > 1:
        # Draw path
        pts = centroids_px.astype(np.int32)
        for i in range(len(pts) - 1):
            cv2.line(path_img, tuple(pts[i]), tuple(pts[i+1]), (0, 0, 255), 2)
        # Draw start point (green)
        cv2.circle(path_img, tuple(pts[0]), 5, (0, 255, 0), -1)
        # Draw end point (red)
        cv2.circle(path_img, tuple(pts[-1]), 5, (0, 0, 255), -1)
    cv2.imwrite(str(output_dir / "tracked_path.png"), path_img)
    
    # 3. features_count.png (plot features vs frame index)
    n_features = status_info['n_features_per_frame']
    frame_indices = np.arange(len(n_features))
    
    plt.figure(figsize=(10, 6))
    plt.plot(frame_indices, n_features, 'b-', linewidth=1.5)
    plt.xlabel('Frame Index')
    plt.ylabel('Number of Features')
    plt.title('Feature Count Over Time')
    plt.grid(True, alpha=0.3)
    
    # Mark reseed frames
    if status_info['reseed_frames']:
        reseed_indices = np.array(status_info['reseed_frames'])
        reseed_features = [n_features[i] for i in reseed_indices if i < len(n_features)]
        if reseed_features:
            plt.scatter(reseed_indices[:len(reseed_features)], reseed_features, 
                       c='r', marker='x', s=100, label='Re-seed', zorder=5)
            plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "features_count.png", dpi=150)
    plt.close()
    
    # 4. centroids.csv (time, x_px, y_px, x_sim, y_sim)
    csv_path = output_dir / "centroids.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time_sec', 'x_px', 'y_px', 'x_sim', 'y_sim'])
        for i in range(len(times_sec)):
            writer.writerow([
                times_sec[i],
                centroids_px[i, 0],
                centroids_px[i, 1],
                centroids_sim[i, 0],
                centroids_sim[i, 1]
            ])
