# cp1/src/analysis.py

import matplotlib.pyplot as plt
import numpy as np
import os
from .features import normalize_features
from .clustering_scratch import KMeansScratch
from .clustering_lib import cluster_sklearn

def plot_trajectory(tracks, output_path):
    plt.figure(figsize=(10, 6))
    for obj_id, track in tracks.items():
        positions = np.array(track)
        plt.plot(positions[:, 0], positions[:, 1], label=f'Obj {obj_id}')
        plt.scatter(positions[0, 0], positions[0, 1], marker='o') # Start
        plt.scatter(positions[-1, 0], positions[-1, 1], marker='x') # End
    
    plt.title("Object Trajectories (Pixel Coordinates)")
    plt.xlabel("X (px)")
    plt.ylabel("Y (px)")
    plt.gca().invert_yaxis() # Image coordinates
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def plot_kinematics(time, speed, accel, obj_id, output_dir):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(time, speed)
    plt.title(f"Speed vs Time (Obj {obj_id})")
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (px/s)")
    
    plt.subplot(1, 2, 2)
    plt.plot(time, accel)
    plt.title(f"Accel vs Time (Obj {obj_id})")
    plt.xlabel("Time (s)")
    plt.ylabel("Accel (px/s^2)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"kinematics_obj_{obj_id}.png"))
    plt.close()

def run_clustering_experiments(feature_matrix, output_dir):
    """
    Runs clustering with different norms and saves results.
    """
    if len(feature_matrix) < 2:
        print("Not enough objects for clustering.")
        return

    # Normalize features
    X_norm = normalize_features(feature_matrix)
    
    norms = ['L2', 'L1', 'Linf', 'WeightedL2']
    weights = [1, 1, 2, 2, 5, 5] # Emphasize jerk/jounce
    
    results_txt = []
    
    for norm in norms:
        kmeans = KMeansScratch(k=min(2, len(feature_matrix)), norm=norm, weights=weights)
        labels = kmeans.fit(X_norm)
        
        # Plot
        plt.figure(figsize=(6, 6))
        # Project to 2D (Speed vs Acceleration approx for viz)
        plt.scatter(X_norm[:, 0], X_norm[:, 2], c=labels, cmap='viridis', s=100)
        plt.title(f"Clustering with {norm} Norm")
        plt.xlabel("Norm. Mean Speed")
        plt.ylabel("Norm. Mean Accel")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"cluster_{norm}.png"))
        plt.close()
        
        results_txt.append(f"Norm {norm}: Labels {labels}")

    # Sklearn comparison
    labels_sk, _ = cluster_sklearn(X_norm, k=min(2, len(feature_matrix)))
    results_txt.append(f"Sklearn (L2): Labels {labels_sk}")
    
    with open(os.path.join(output_dir, "clustering_results.txt"), "w") as f:
        f.write("\n".join(results_txt))

