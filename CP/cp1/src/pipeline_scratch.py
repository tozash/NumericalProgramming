# cp1/src/pipeline_scratch.py

import argparse
import os
import numpy as np
import cv2
from .config import *
from .video_io import read_video_frames
from .background_model import BackgroundModelScratch
from .object_tracking import find_centroids_scratch, Tracker
from .features import extract_features
from .analysis import plot_trajectory, plot_kinematics, run_clustering_experiments

def main():
    parser = argparse.ArgumentParser(description="Scratch Pipeline")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/scratch")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # 1. Process Video
    bg_model = BackgroundModelScratch(alpha=BG_ALPHA, threshold=BG_THRESHOLD)
    tracker = Tracker()
    
    frame_idx = 0
    for frame in read_video_frames(args.video, resize_dim=(RESIZE_WIDTH, RESIZE_HEIGHT)):
        # Detection
        mask = bg_model.apply(frame)
        centroids = find_centroids_scratch(mask, min_area=MIN_AREA)
        
        # Tracking
        tracker.update(centroids)
        
        # Visualization (Optional: save first few frames)
        if frame_idx < 100 and frame_idx % 10 == 0:
            vis = frame.copy()
            # Draw existing tracks
            for obj_id, track in tracker.objects.items():
                if len(track) > 1:
                    pts = np.array(track, np.int32)
                    cv2.polylines(vis, [pts], False, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(args.output, f"frame_{frame_idx:04d}.jpg"), vis)
            
        frame_idx += 1
        
    # 2. Compute Derivatives & Features
    all_features = []
    valid_objects = []
    
    for obj_id, positions in tracker.objects.items():
        if len(positions) < 10:
            continue # Ignore short tracks
            
        track_data = {'positions': np.array(positions)}
        
        # Features (includes derivative computation internally)
        feat = extract_features(track_data, TIME_STEP)
        all_features.append(feat)
        valid_objects.append(obj_id)
        
        # Detailed kinematic plots for each object
        # Re-compute to get timeseries for plotting
        # (extract_features computes summaries, but we want plots too)
        # This is a bit redundant but cleaner for code structure
        from .features import compute_kinematics # Import here or use util
        pos_arr = np.array(positions)
        kin = compute_kinematics(pos_arr[:, 0], TIME_STEP) # just X for simplicity or mag
        kin_y = compute_kinematics(pos_arr[:, 1], TIME_STEP)
        
        speed = np.sqrt(kin['velocity']**2 + kin_y['velocity']**2)
        acc = np.sqrt(kin['acceleration']**2 + kin_y['acceleration']**2)
        time = np.arange(len(speed)) * TIME_STEP
        
        plot_kinematics(time, speed, acc, obj_id, args.output)

    plot_trajectory(tracker.objects, os.path.join(args.output, "trajectories.png"))

    # 3. Clustering
    if all_features:
        feature_matrix = np.array(all_features)
        run_clustering_experiments(feature_matrix, args.output)

if __name__ == "__main__":
    main()

