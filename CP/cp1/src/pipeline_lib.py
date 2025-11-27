# cp1/src/pipeline_lib.py

import argparse
import os
import numpy as np
import cv2
from .config import *
from .video_io import read_video_frames
from .background_model import BackgroundModelLib
from .object_tracking import find_centroids_lib, Tracker
from .features import extract_features
from .analysis import plot_trajectory, plot_kinematics, run_clustering_experiments

def main():
    parser = argparse.ArgumentParser(description="Library Pipeline")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/lib")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # 1. Process Video (Library BG Subtraction)
    bg_model = BackgroundModelLib(threshold=BG_THRESHOLD)
    tracker = Tracker() # Reuse tracker logic as allowed
    
    frame_idx = 0
    for frame in read_video_frames(args.video, resize_dim=(RESIZE_WIDTH, RESIZE_HEIGHT)):
        mask = bg_model.apply(frame)
        centroids = find_centroids_lib(mask, min_area=MIN_AREA)
        tracker.update(centroids)
        frame_idx += 1
        
    # 2. Analysis (Shared logic)
    all_features = []
    
    for obj_id, positions in tracker.objects.items():
        if len(positions) < 10: continue
            
        track_data = {'positions': np.array(positions)}
        feat = extract_features(track_data, TIME_STEP)
        all_features.append(feat)
        
        # Kinematics for plot
        from .derivatives import compute_kinematics
        pos_arr = np.array(positions)
        kin_x = compute_kinematics(pos_arr[:, 0], TIME_STEP)
        kin_y = compute_kinematics(pos_arr[:, 1], TIME_STEP)
        speed = np.sqrt(kin_x['velocity']**2 + kin_y['velocity']**2)
        acc = np.sqrt(kin_x['acceleration']**2 + kin_y['acceleration']**2)
        time = np.arange(len(speed)) * TIME_STEP
        plot_kinematics(time, speed, acc, obj_id, args.output)

    plot_trajectory(tracker.objects, os.path.join(args.output, "trajectories.png"))

    if all_features:
        feature_matrix = np.array(all_features)
        run_clustering_experiments(feature_matrix, args.output)

if __name__ == "__main__":
    main()

